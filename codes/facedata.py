import warnings

# Suppress torchvision warning
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.utils.data import Dataset


ROOT_TRAIN = "/media/jag/volD2/face_dummy"
ROOT_VAL = "/media/jag/volD2/face_dummy"

class SafeColorJitter(transforms.ColorJitter):
    """ColorJitter without hue to avoid issues"""
    def __init__(self, brightness=0, contrast=0, saturation=0):
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=0)

def get_dynamic_loader(class_range=(0, 49), mode="train", batch_size=32, image_size=112, num_workers=0, train_split=0.85):
    data_dir = ROOT_TRAIN

    if mode == "train":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            SafeColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
    
    # Split indices into train/val
    torch.manual_seed(42)
    total_indices = len(indices)
    train_size = int(train_split * total_indices)
    perm = torch.randperm(total_indices)
    
    if mode == "train":
        split_indices = [indices[i] for i in perm[:train_size]]
    else:
        split_indices = [indices[i] for i in perm[train_size:]]
    
    subset_dataset = Subset(dataset, split_indices)

    loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),  
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False
    )

    return loader

def get_class_info(data_dir=ROOT_TRAIN):
    """Get dataset class information"""
    dataset = datasets.ImageFolder(root=data_dir)
    return {
        'num_classes': len(dataset.classes),
        'class_names': dataset.classes,
        'class_to_idx': dataset.class_to_idx
    }

# Test dataset
if __name__ == "__main__":
    # Test class info
    class_info = get_class_info()
    print(f"Found {class_info['num_classes']} classes")
    print(f"Sample classes: {class_info['class_names'][:10]}")
    
    # Test dataloader
    train_loader = get_dynamic_loader(class_range=(0, 4), mode="train", batch_size=8)
    
    # Test batch
    for images, labels in train_loader:
        print(f"Train batch shape: {images.shape}")
        print(f"Labels: {labels}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        break