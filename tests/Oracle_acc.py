import os
import json
import torch
from torch import nn
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score

from codes.utils import get_model
from codes.data import get_dynamic_loader


# ----------------- Evaluation Function -----------------

def evaluate_with_per_class(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    all_preds = []
    all_labels = []

    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for pred, true in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                per_class_total[true] += 1
                if pred == true:
                    per_class_correct[true] += 1

    avg_loss = total_loss / len(dataloader)
    overall_acc = accuracy_score(all_labels, all_preds)

    # Compute per-class accuracy for all classes 0–99
    per_class_acc = {}
    for class_id in range(100):
        correct = per_class_correct[class_id]
        total = per_class_total[class_id]
        acc = (correct / total) * 100 if total > 0 else 0.0
        per_class_acc[str(class_id)] = round(acc, 2)

    return avg_loss, round(overall_acc * 100, 2), per_class_acc


# ----------------- Oracle Accuracy Evaluation -----------------

def evaluate_oracle_checkpoints():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("./results", exist_ok=True)

    class_ranges = [(0, 49), (10, 59), (20, 69), (30, 79), (40, 89), (50, 99)]

    print("\n🔍 Oracle Accuracy Evaluation with Per-Class Results\n")

    for class_start, class_end in class_ranges:
        print(f"🔹 Evaluating Oracle Model: Classes {class_start}–{class_end}")

        checkpoint_path = f"./checkpoints/oracle/{class_start}_{class_end}.pth"
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            continue

        # Model always outputs 100 classes
        model = get_model(num_classes=100, use_lora=True, pretrained=False)
        model.to(device)

        state_dict = torch.load(checkpoint_path, map_location=device,weights_only=True)
        model.load_state_dict(state_dict,strict=False)

        # Load val data for this class range
        val_loader = get_dynamic_loader(class_range=(class_start, class_end), mode="val", batch_size=64)

        # Evaluate
        val_loss, overall_acc, per_class_acc = evaluate_with_per_class(model, val_loader, device)

        print(f"✅ Val Loss: {val_loss:.4f} | Overall Acc (on range): {overall_acc:.2f}%")

        # Save results to JSON
        results = {
            "overall_acc": overall_acc,
            "per_class_acc": per_class_acc
        }

        result_path = f"./results/oracle/{class_start}_{class_end}.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"📁 Results saved to {result_path}\n")

    print("🏁 Evaluation complete for all models.")


# ----------------- Main -----------------

if __name__ == "__main__":
    evaluate_oracle_checkpoints()
