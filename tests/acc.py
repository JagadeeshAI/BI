# File: tests/acc.py

import os
import torch
from codes.data import get_dynamic_loader
from codes.utils import get_model, load_model_weights
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def main():
    # ─── CONFIGURE YOUR BLOCK ────────────────────────────────────────────────────
    # Change this to one of your trained ranges:
    #   (0,49), (10,59), (20,69), (30,79), (40,89), or (50,99)
    class_start, class_end = 10, 59
    # ────────────────────────────────────────────────────────────────────────────

    # 1) device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2) build the same VisionTransformer you used in Oraclefinetune.py
    #    (no LoRA, 100‐way head)
    model = get_model(num_classes=100, use_lora=False, pretrained=False)
    model.to(device)

    # 3) load the corresponding oracle checkpoint
    ckpt_name = f"oracle_{class_start}_{class_end}.pth"
    ckpt_path = os.path.join(os.getcwd(), "checkpoints", ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Could not find checkpoint:\n  {ckpt_path}")
    print(f"Loading weights from {ckpt_path}")
    load_model_weights(model, ckpt_path, strict=False)

    # 4) prepare a val loader for just that slice of classes
    val_loader = get_dynamic_loader(
        class_range=(class_start, class_end),
        mode="val",
        batch_size=64
    )

    # 5) run evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)                     # -> [B,100]
            preds = outputs.argmax(dim=1)             # global class IDs 0–99
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Oracle accuracy on classes {class_start}–{class_end}: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
