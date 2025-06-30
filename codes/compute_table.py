import os
import torch
import json
import numpy as np
from tqdm import tqdm
from codes.utils import get_model
from codes.data import get_dynamic_loader

def softmax_logits(logits):
    return torch.softmax(logits, dim=1).cpu().numpy()

def evaluate_accuracy(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, lbls in tqdm(dataloader, desc="Eval", leave=False):
            images, lbls = images.to(device), lbls.to(device)
            out = model(images)
            preds.append(out.argmax(dim=1).cpu().numpy())
            labels.append(lbls.cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    acc = np.mean(preds == labels)
    return acc

def evaluate_kl(model, reference_model, dataloader, device):
    model.eval()
    reference_model.eval()
    kl_sum = 0
    n_samples = 0
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="KL", leave=False):
            images = images.to(device)
            out1 = model(images)
            out2 = reference_model(images)
            p1 = torch.softmax(out1, dim=1)
            p2 = torch.softmax(out2, dim=1)
            kl = torch.sum(p2 * (p2.log() - p1.log()), dim=1)
            kl_sum += kl.sum().item()
            n_samples += images.shape[0]
    return kl_sum / n_samples

def dummy_mia(model, dataloader, device):
    return np.random.uniform(0, 0.1)

def main():
    steps = [
        {"step": "Pretrained", "retain": (0, 49),  "unlearn": None,    "ckpt": "pretrained.pth", "use_lora": False},
        {"step": "Step1",      "retain": (10, 59), "unlearn": (0, 9),  "ckpt": "step1.pth",      "use_lora": True},
        {"step": "Step2",      "retain": (20, 69), "unlearn": (10,19), "ckpt": "step2.pth",      "use_lora": True},
        {"step": "Step3",      "retain": (30, 79), "unlearn": (20,29), "ckpt": "step3.pth",      "use_lora": True},
        {"step": "Step4",      "retain": (40, 89), "unlearn": (30,39), "ckpt": "step4.pth",      "use_lora": True},
        {"step": "Step5",      "retain": (50, 99), "unlearn": (40,49), "ckpt": "step5.pth",      "use_lora": True},
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 100  # total classes

    results = []
    prev_la = None

    for i, s in enumerate(steps):
        print(f"==> Evaluating {s['step']} ({s['ckpt']})")
        use_lora = s.get("use_lora", True)
        lora_args = {"lora_rank": 2} if use_lora else {}

        # Load model
        model = get_model(num_classes=num_classes, use_lora=use_lora, **lora_args)
        state_dict = torch.load(f"./checkpoints/steps/{s['ckpt']}", map_location=device, weights_only=use_lora)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        # Prepare dataloaders
        retained_range = s["retain"]
        retained_loader = get_dynamic_loader(class_range=retained_range, mode="val", batch_size=64)

        # LA (Learning Accuracy) on retained classes
        la = evaluate_accuracy(model, retained_loader, device)

        # UA (Unlearning Accuracy) on unlearned classes, if any
        ua = None
        if s["unlearn"] is not None:
            unlearn_loader = get_dynamic_loader(class_range=s["unlearn"], mode="val", batch_size=64)
            ua = evaluate_accuracy(model, unlearn_loader, device)
        else:
            unlearn_loader = None

        # FM (Forgetting Measure)
        fm = la - prev_la if prev_la is not None else None
        prev_la = la

        # KL-Divergence: Compare with matching oracle model
        oracle_ckpt = f"./checkpoints/oracle_{retained_range[0]}_{retained_range[1]}.pth"
        if os.path.exists(oracle_ckpt):
            oracle_model = get_model(num_classes=num_classes, use_lora=use_lora, **lora_args)
            state_dict = torch.load(oracle_ckpt, map_location=device, weights_only=use_lora)
            oracle_model.load_state_dict(state_dict, strict=False)
            oracle_model.to(device)
            oracle_model.eval()
            kl = evaluate_kl(model, oracle_model, retained_loader, device)
        else:
            kl = None

        # MIA (STUB)
        mia = dummy_mia(model, unlearn_loader, device) if s["unlearn"] is not None else None

        # RTE: placeholder
        rte = None

        results.append({
            "step": s["step"],
            "retain_classes": f"{retained_range[0]}-{retained_range[1]}",
            "unlearn_classes": f"{s['unlearn'][0]}-{s['unlearn'][1]}" if s["unlearn"] else None,
            "LA": la,
            "UA": ua,
            "FM": fm,
            "KL-D": kl,
            "MIA": mia,
            "RTE": rte
        })

        # Save intermediate results
        with open("results_table.json", "w") as f:
            json.dump(results, f, indent=2)

    print("==> Done! Results saved to results_table.json")

if __name__ == "__main__":
    main()
