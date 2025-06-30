# codes/finetune_imagenet.py

import argparse
import os
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


def mixup_data(x, y, alpha=0.2, device='cuda'):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_one_epoch(model, loader, criterion, optimizer, device, mixup_alpha):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Train"):
        images, labels = images.to(device), labels.to(device)
        # mixup?
        if mixup_alpha > 0:
            images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha, device)
            outputs = model(images)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validate"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    return running_loss / len(loader.dataset), correct / len(loader.dataset)


def main():
    p = argparse.ArgumentParser(description="Fine-tune ViT or Swin on ImageNet")
    p.add_argument("--data-dir",        type=str,   required=True, help="root of ImageNet (train/ & val/ subfolders)")
    p.add_argument("--model",           type=str,   default="vit_base_patch16_224", 
                   help="timm model: vit_base_patch16_224 or swin_tiny_patch4_window7_224")
    p.add_argument("--batch-size",      type=int,   default=256)
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight-decay",    type=float, default=0.05)
    p.add_argument("--mixup-alpha",     type=float, default=0.2, 
                   help="MixUp alpha; 0 to disable")
    p.add_argument("--workers",         type=int,   default=8)
    p.add_argument("--output-dir",      type=str,   default="./checkpoints")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # ----- data transforms -----
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4,0.4,0.4,0.1),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
        transforms.RandomErasing(p=0.5, scale=(0.02,0.2), ratio=(0.3,3.3)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(args.data_dir,   "val"), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # ----- model -----
    model = timm.create_model(args.model, pretrained=True, num_classes=1000)
    model.to(device)

    # ----- optimizer & sched -----
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ----- loss -----
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss_train = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device, args.mixup_alpha)
        loss_val, acc_val = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"\nEpoch {epoch}/{args.epochs}  "
              f"Train Loss: {loss_train:.4f}  "
              f"Val Loss: {loss_val:.4f}  "
              f"Val Acc: {acc_val*100:.2f}%  "
              f"Time: {time.time()-start:.0f}s")

        ckpt = os.path.join(args.output_dir, f"{args.model}_best.pth")
        if acc_val > best_acc:
            best_acc = acc_val
            torch.save(model.state_dict(), ckpt)
            print(f"→ New best ({best_acc*100:.2f}%), saved to {ckpt}")

    print("Finished fine-tuning.")


if __name__ == "__main__":
    main()
