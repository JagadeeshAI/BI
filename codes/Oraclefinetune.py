import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from codes.utils import get_model
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


# ----------------- Oracle Training Loop -----------------

def train_oracle_model(class_start, class_end):
    class_range = (class_start, class_end)
    print(f"\n🚀 Training Oracle Model on Classes: {class_start}–{class_end}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 100  # Fixed

    model = get_model(num_classes=num_classes, use_lora=False, pretrained=True)
    model.to(device)

    # Data
    train_loader = get_dynamic_loader(class_range=class_range, mode="train", batch_size=64)
    val_loader = get_dynamic_loader(class_range=class_range, mode="val", batch_size=64)

    # Training config
    num_epochs = 10
    lr = 3e-4
    weight_decay = 0.1
    label_smoothing = 0.1

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\n📅 Epoch {epoch}/{num_epochs} — Class Range: {class_start}-{class_end}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"📊 Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc * 100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"./checkpoints/oracle_{class_start}_{class_end}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best Val Acc: {val_acc * 100:.2f}% — Model saved to {save_path}")
        else:
            print(f"No improvement. Best so far: {best_val_acc * 100:.2f}%")

        scheduler.step()

    print("🏁 Training finished.")


# ----------------- Main -----------------

def main():
    os.makedirs("./checkpoints", exist_ok=True)

    class_ranges = [(0, 49), (10, 59), (20, 69), (30, 79), (40, 89), (50, 99)]

    for start, end in class_ranges:
        train_oracle_model(start, end)


if __name__ == "__main__":
    main()
