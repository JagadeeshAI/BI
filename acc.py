import torch
import torch.nn as nn
import json
from tqdm import tqdm
from collections import defaultdict

from utils import get_model, load_model_weights  # ⬅️ new helper from updated utils.py
from data import get_dynamic_loader


def evaluate(model, dataloader, device, num_classes):
    model.eval()
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="🔍 Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            for label, pred in zip(labels, preds):
                class_total[label.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1

    # Compute per-class accuracy
    class_acc = {}
    for cls in range(num_classes):
        total = class_total[cls]
        correct = class_correct[cls]
        acc = 100.0 * correct / total if total > 0 else 0.0
        class_acc[cls] = round(acc, 2)

    overall_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    return overall_acc, class_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = 60  # ✅ We are evaluating 0–59 classes
    model = get_model(num_classes=num_classes, use_lora=True, lora_rank=2, pretrained=False).to(device)

    # ✅ Safely load partial weights (ignoring mismatches like head/lora)
    load_model_weights(model, "0_59.pth", strict=False)

    # ✅ Use full 0–59 range
    val_loader = get_dynamic_loader(class_range=(0, 59), mode="val", batch_size=64)

    overall_acc, class_acc = evaluate(model, val_loader, device, num_classes=100)

    results = {
        "overall_accuracy": round(overall_acc, 2),
        "class_wise_accuracy": class_acc
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Evaluation completed!")
    print(f"🔢 Overall Accuracy: {overall_acc:.2f}%")
    print("📊 Class-wise Accuracy:")
    for k in sorted(class_acc):
        print(f"  Class {k:02d}: {class_acc[k]:.2f}%")


if __name__ == "__main__":
    main()
