import os
import torch
import json
from tqdm import tqdm
from codes.utils import get_model, load_model_weights
from codes.data import get_dynamic_loader

# ✅ Step-wise ranges (customizable)
STEP_RANGES = {
    0: {'forget': (0, 49), 'retain': (49, 49), 'new': (50, 50), 'overall': (00, 49)},
    1: {'forget': (0, 9), 'retain': (10, 49), 'new': (50, 59), 'overall': (10, 59)},
    2: {'forget': (0, 19), 'retain': (20, 59), 'new': (60, 69), 'overall': (20, 69)},
    3: {'forget': (0, 29), 'retain': (30, 69), 'new': (70, 79), 'overall': (30, 79)},
    4: {'forget': (0, 39), 'retain': (40, 79), 'new': (80, 89), 'overall': (40, 89)},
    5: {'forget': (0, 49), 'retain': (50, 89), 'new': (90, 99), 'overall': (50, 99)}
}


def evaluate(model, dataloader, device, desc="Evaluating"):
    """Evaluate model on given dataloader, return accuracy (%)"""
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=desc, leave=False, unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def evaluate_checkpoint(checkpoint_path, step,  batch_size=64):
    """Load checkpoint and evaluate on forget/retain/new/overall ranges"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Evaluating {checkpoint_path} on STEP {step} (device={device})")

    # Load model
    model = get_model(num_classes=100).to(device)
    load_model_weights(model, checkpoint_path, strict=False)
    model.eval()

    results = {}
    for eval_type, class_range in STEP_RANGES[step].items():
        print(f"\n🔍 Evaluating {eval_type} classes: {class_range[0]}–{class_range[1]}")
        loader = get_dynamic_loader(
            class_range=class_range,
            mode="val",
            batch_size=batch_size,
            image_size=224,
            num_workers=0
        )
        acc = evaluate(model, loader, device, desc=f"🔍 {eval_type} {class_range}")
        results[eval_type] = round(acc, 2)
        print(f"  ✅ {eval_type:<8} ({class_range[0]}–{class_range[1]}): {acc:.2f}%")

    # Save results JSON
    out_file = f"results_step{step}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved results to {out_file}")

    return results



if __name__ == "__main__":
    # ✅ Example usage — change these two lines
    checkpoint = "checkpoints/oracle/0_49.pth"
    step = 0

    results = evaluate_checkpoint(checkpoint, step)
