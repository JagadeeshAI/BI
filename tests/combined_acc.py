import os
import json
import torch
from tqdm import tqdm
from collections import defaultdict
from codes.utils import get_model
from data import get_dynamic_loader

def evaluate_model(model, dataloader, device, num_classes):
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

    class_acc = {
        str(cls): round(100.0 * class_correct[cls] / class_total[cls], 2)
        if class_total[cls] > 0 else 0.0
        for cls in range(num_classes)
    }

    overall_acc = round(100.0 * total_correct / total_samples, 2)
    return overall_acc, class_acc

def load_weights_safely(model, path):
    state_dict = torch.load(path, map_location="cpu")
    filtered = {
        k: v for k, v in state_dict.items()
        if k in model.state_dict() and model.state_dict()[k].shape == v.shape
    }
    model.load_state_dict(filtered, strict=False)

def evaluate_and_save(model_path, class_range, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 100
    model = get_model(num_classes=num_classes, use_lora=True, pretrained=False).to(device)
    load_weights_safely(model, model_path)

    val_loader = get_dynamic_loader(class_range=class_range, mode="val", batch_size=64)
    overall_acc, class_acc = evaluate_model(model, val_loader, device, num_classes)

    results = {
        "model_path": model_path,
        "evaluated_classes": f"{class_range[0]}-{class_range[1]}",
        "overall_accuracy": overall_acc,
        "class_wise_accuracy": class_acc
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ {model_path} evaluated -> {output_path}")

if __name__ == "__main__":
    evaluate_and_save("0_49.pth", (0, 49), "results_0_49.json")
    evaluate_and_save("0_59.pth", (0, 59), "results_0_59.json")
    evaluate_and_save("forget_0_9.pth", (10, 59), "results_10_59.json")
