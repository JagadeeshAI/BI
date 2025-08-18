import os
import torch
from torchvision import datasets, transforms  
from torch.utils.data import DataLoader, Subset
import random

# Hardcoded root directories
ROOT_TRAIN = "/media/jag/volD2/cifer100/cifer/train"
ROOT_VAL = "/media/jag/volD2/cifer100/cifer/val"

class SafeColorJitter:
    """Safer ColorJitter implementation that avoids overflow errors"""
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        # Only use brightness, contrast, and saturation - skip hue to avoid overflow
        self.brightness = brightness
        self.contrast = contrast  
        self.saturation = saturation
        self.color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)
    
    def __call__(self, img):
        return self.color_jitter(img)


def get_dynamic_loader(class_range=(0, 99), mode="train", batch_size=32, image_size=224, num_workers=0, data_percentage=1.0):
    data_dir = ROOT_TRAIN if mode == "train" else ROOT_VAL

    if mode == "train":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            SafeColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Removed hue parameter entirely
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)), 
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    start_class, end_class = class_range
    allowed_classes = list(range(start_class, end_class + 1))

    indices = [i for i, (_, label) in enumerate(dataset.samples) if label in allowed_classes]
    
    # Apply data percentage sampling
    if data_percentage < 1.0:
        num_samples = int(len(indices) * data_percentage)
        indices = random.sample(indices, num_samples)
    
    subset_dataset = Subset(dataset, indices)

    loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),  
        num_workers=num_workers,  # Keep at 0 to avoid multiprocessing issues
        pin_memory=False,  # Keep False to reduce memory issues
        persistent_workers=False  # Added to prevent worker issues
    )

    return loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        train_loader = get_dynamic_loader(class_range=(0, 49), mode="train")
        val_loader = get_dynamic_loader(class_range=(0, 49), mode="val")

        train_images, train_labels = next(iter(train_loader))
        val_images, val_labels = next(iter(val_loader))

        print(f"[Train] Batch image shape: {train_images.shape}, Labels shape: {train_labels.shape}")
        print(f"[Val] Batch image shape: {val_images.shape}, Labels shape: {val_labels.shape}")
        
        # Print some additional info for debugging
        print(f"Train labels range: {train_labels.min().item()} to {train_labels.max().item()}")
        print(f"Val labels range: {val_labels.min().item()} to {val_labels.max().item()}")
        
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {e}")
        print("Try reducing batch_size or using different augmentation parameters")


if __name__ == "__main__":
    main()