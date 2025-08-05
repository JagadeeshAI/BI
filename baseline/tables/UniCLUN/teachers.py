import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from codes.utils import get_model, load_model_weights
from codes.data import get_dynamic_loader

# ----------------- Training & Evaluation Functions -----------------

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


# ----------------- Teacher Training Loop -----------------

def train_teacher_model(class_ranges, teacher_type, save_name):
    """
    Train a teacher model on specified class ranges
    
    Args:
        class_ranges: list of tuples [(start1, end1), (start2, end2), ...]
        teacher_type: "good" or "bad"
        save_name: filename for saving the model
    """
    print(f"\n🚀 Training {teacher_type.upper()} Teacher: {save_name}")
    print(f"📚 Training on class ranges: {class_ranges}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 100

    model = get_model(num_classes=num_classes, use_lora=False, pretrained=True)
    model.to(device)

    # Data - create separate loaders for each range and combine
    all_train_loaders = []
    all_val_loaders = []
    
    for class_range in class_ranges:
        train_loader = get_dynamic_loader(class_range=class_range, mode="train", batch_size=64)
        val_loader = get_dynamic_loader(class_range=class_range, mode="val", batch_size=32)
        all_train_loaders.append(train_loader)
        all_val_loaders.append(val_loader)
    
    # Combine datasets
    from torch.utils.data import ConcatDataset
    
    train_datasets = [loader.dataset for loader in all_train_loaders]
    val_datasets = [loader.dataset for loader in all_val_loaders]
    
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_val_dataset = ConcatDataset(val_datasets)
    
    train_loader = DataLoader(combined_train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(combined_val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Training config
    num_epochs = 40
    lr = 3e-4
    weight_decay = 0.1
    label_smoothing = 0.1

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\n📅 Epoch {epoch}/{num_epochs} — {teacher_type.upper()} Teacher: {save_name}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"📊 Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc * 100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"./baseline/teachers/{save_name}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best Val Acc: {val_acc * 100:.2f}% — Model saved to {save_path}")
        else:
            print(f"No improvement. Best so far: {best_val_acc * 100:.2f}%")

        scheduler.step()

    print("🏁 Training finished.")


# ----------------- Main -----------------

def main():
    os.makedirs("./baseline/teachers", exist_ok=True)

    # Define all teacher configurations
    teacher_configs = [
        # Student 10-59
        {
            "good_ranges": [(10, 59)],
            "bad_ranges": [(0, 9), (60, 99)],
            "student_name": "student_10_59"
        },
        # Student 20-69
        {
            "good_ranges": [(20, 69)],
            "bad_ranges": [(0, 19), (70, 99)],
            "student_name": "student_20_69"
        },
        # Student 30-79
        {
            "good_ranges": [(30, 79)],
            "bad_ranges": [(0, 29), (80, 99)],
            "student_name": "student_30_79"
        },
        # Student 40-89
        {
            "good_ranges": [(40, 89)],
            "bad_ranges": [(0, 39), (90, 99)],
            "student_name": "student_40_89"
        },
        # Student 50-99
        {
            "good_ranges": [(50, 99)],
            "bad_ranges": [(0, 49)],
            "student_name": "student_50_99"
        }
    ]

    # Train all teachers
    for config in teacher_configs:
        student_name = config["student_name"]
        
        # Train Good Teacher
        good_teacher_name = f"{student_name}_good_teacher"
        train_teacher_model(
            class_ranges=config["good_ranges"],
            teacher_type="good",
            save_name=good_teacher_name
        )
        
        # Train Bad Teacher
        bad_teacher_name = f"{student_name}_bad_teacher"
        train_teacher_model(
            class_ranges=config["bad_ranges"],
            teacher_type="bad",
            save_name=bad_teacher_name
        )

    print("\n🎉 All teachers trained successfully!")
    print("📁 Teachers saved in: ./baseline/teachers/")


if __name__ == "__main__":
    main()