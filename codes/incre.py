import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from collections import defaultdict
from data import get_dynamic_loader
from codes.utils import get_model

def evaluate(model, dataloader, device, num_classes):
    model.eval()
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            for label, pred in zip(labels, preds):
                class_total[label.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1

    class_acc = {}
    for cls in range(num_classes):
        total = class_total[cls]
        correct = class_correct[cls]
        acc = 100.0 * correct / total if total > 0 else 0.0
        class_acc[cls] = round(acc, 2)

    overall_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    return overall_acc, class_acc

def train_one_epoch(model, loader_new, loader_old, optimizer, criterion, device, alpha=0.5):
    model.train()
    total_loss = 0
    old_iter = iter(loader_old)

    for images_new, labels_new in tqdm(loader_new, desc="Training"):
        images_new, labels_new = images_new.to(device), labels_new.to(device)

        try:
            images_old, labels_old = next(old_iter)
        except StopIteration:
            old_iter = iter(loader_old)
            images_old, labels_old = next(old_iter)

        images_old, labels_old = images_old.to(device), labels_old.to(device)

        outputs_new = model(images_new)
        outputs_old = model(images_old)

        loss_new = criterion(outputs_new, labels_new)
        loss_old = criterion(outputs_old, labels_old)
        loss = alpha * loss_old + (1 - alpha) * loss_new

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader_new)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = 60
    model = get_model(num_classes=num_classes, use_lora=True, pretrained=False, lora_rank=2).to(device)
    state_dict = torch.load("0_49.pth", map_location=device)

    # Filter out incompatible keys (LoRA or head size mismatch)
    filtered_state_dict = {k: v for k, v in state_dict.items()
                        if k in model.state_dict() and model.state_dict()[k].shape == v.shape}

    missing_keys = [k for k in model.state_dict() if k not in filtered_state_dict]
    print(f"⚠️ Skipping incompatible or missing keys: {missing_keys[:5]}... (+{len(missing_keys) - 5} more)" if len(missing_keys) > 5 else f"⚠️ Skipping keys: {missing_keys}")

    model.load_state_dict(filtered_state_dict, strict=False)


    loader_new = get_dynamic_loader(class_range=(50, 59), mode="train", batch_size=64)
    loader_old = get_dynamic_loader(class_range=(0, 49), mode="train", batch_size=64)

    val_loader_0_49 = get_dynamic_loader(class_range=(0, 49), mode="val", batch_size=64)
    val_loader_50_59 = get_dynamic_loader(class_range=(50, 59), mode="val", batch_size=64)
    val_loader_full = get_dynamic_loader(class_range=(0, 59), mode="val", batch_size=64)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    num_epochs = 10

    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_one_epoch(model, loader_new, loader_old, optimizer, criterion, device)

        acc_0_49, _ = evaluate(model, val_loader_0_49, device, num_classes)
        acc_50_59, _ = evaluate(model, val_loader_50_59, device, num_classes)

        print(f"📈 Train Loss: {train_loss:.4f}")
        print(f"🔍 Val Acc (0–49): {acc_0_49:.2f}% | Val Acc (50–59): {acc_50_59:.2f}%")

    # Final combined evaluation
    overall_acc, class_acc = evaluate(model, val_loader_full, device, num_classes)
    torch.save(model.state_dict(), "0_59.pth")

    results = {
        "overall_accuracy": round(overall_acc, 2),
        "class_wise_accuracy": class_acc
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Training completed. Final Overall Accuracy: {overall_acc:.2f}%")

if __name__ == "__main__":
    main()
