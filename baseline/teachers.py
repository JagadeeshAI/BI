import torch
import torch.nn as nn
import torch.optim as optim
from timm import create_model
from tqdm import tqdm
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained DeiT model and adapt for fine-tuning
def load_model(num_classes):
    model = create_model('deit_tiny_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, 100)
    return model.to(device)

# Training loop for one epoch
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    loop = tqdm(dataloader, desc="Train", leave=False)
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# Evaluation loop
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    loop = tqdm(dataloader, desc="Val", leave=False)
    with torch.no_grad():
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# Main training loop
def finetune_deit(train_loader, val_loader, num_classes, save_path, epochs=10, lr=3e-4):
    model = load_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved checkpoint: {save_path}")

    return model

# Run two trainings with different class ranges
if __name__ == "__main__":
    from codes.data import get_dynamic_loader  # Replace with your actual module

    # First run: class_range 10–49
    # class_range_1 = list(range(10, 50))
    class_range_1 = (10,50)
    train_loader_1 = get_dynamic_loader(class_range=class_range_1, mode="train", batch_size=64)
    val_loader_1 = get_dynamic_loader(class_range=class_range_1, mode="val", batch_size=64)

    checkpoint_path_1 = "baseline/teachers/checkpoint_deit_10_49.pth"
    print("\n🚀 Fine-tuning model for classes 10–49")
    finetune_deit(train_loader_1, val_loader_1, num_classes=len(class_range_1), save_path=checkpoint_path_1)

    # Second run: class_range 10–59
    class_range_2 = (10,60)
    train_loader_2 = get_dynamic_loader(class_range=class_range_2, mode="train", batch_size=64)
    val_loader_2 = get_dynamic_loader(class_range=class_range_2, mode="val", batch_size=64)

    checkpoint_path_2 = "baseline/teachers/checkpoint_deit_10_59.pth"
    print("\n🚀 Fine-tuning model for classes 10–59")
    finetune_deit(train_loader_2, val_loader_2, num_classes=len(class_range_2), save_path=checkpoint_path_2)
