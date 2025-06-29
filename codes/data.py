import os
import torch
from torchvision import datasets, transforms  # ✅ Fixed import
from torch.utils.data import DataLoader, Subset

# Hardcoded root directories
ROOT_TRAIN = "/media/jag/volD/BID_DATA/imagenet-r-split/train"
ROOT_VAL = "/media/jag/volD/BID_DATA/imagenet-r-split/val"

def get_dynamic_loader(class_range=(0, 99), mode="train", batch_size=32, image_size=224, num_workers=4):
    assert mode in ["train", "val"], "mode must be 'train' or 'val'"
    data_dir = ROOT_TRAIN if mode == "train" else ROOT_VAL

    if mode == "train":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # e.g., 256 for 224 input
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    start_class, end_class = class_range
    allowed_classes = list(range(start_class, end_class + 1))

    indices = [i for i, (_, label) in enumerate(dataset.samples) if label in allowed_classes]
    subset_dataset = Subset(dataset, indices)

    loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),  # ✅ Only shuffle during training
        num_workers=num_workers,
        multiprocessing_context="fork"
    )

    return loader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = get_dynamic_loader(class_range=(0, 49), mode="train")
    val_loader = get_dynamic_loader(class_range=(0, 49), mode="val")

    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))

    print(f"[Train] Batch image shape: {train_images.shape}, Labels shape: {train_labels.shape}")
    print(f"[Val] Batch image shape: {val_images.shape}, Labels shape: {val_labels.shape}")


if __name__ == "__main__":
    main()
