# File: codes/data.py
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Custom Gaussian noise if torchvision version lacks it
class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0., std=0.05):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

ROOT_TRAIN = "/media/jag/volD/BID_DATA/imagenet-r-split/train"
ROOT_VAL   = "/media/jag/volD/BID_DATA/imagenet-r-split/val"

def get_dynamic_loader(class_range=(0, 99), mode="train", batch_size=32, image_size=224, num_workers=4):
    assert mode in ["train", "val"]
    data_dir = ROOT_TRAIN if mode == "train" else ROOT_VAL

    if mode == "train":
        # Stronger and more diverse augmentations to combat overfitting
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.GaussianBlur(kernel_size=(7, 7), sigma=(1.0, 2.0)),
            transforms.ToTensor(),
            # Use ImageNet's recommended mean/std for normalization
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            AddGaussianNoise(mean=0.0, std=0.05),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    start_class, end_class = class_range
    allowed = set(range(start_class, end_class + 1))
    indices = [i for i, (_, label) in enumerate(dataset.samples) if label in allowed]
    subset = Subset(dataset, indices)

    loader = DataLoader(subset,
                        batch_size=batch_size,
                        shuffle=(mode == "train"),
                        num_workers=num_workers,
                        multiprocessing_context="fork")
    return loader

# The rest of your data loading code remains unchanged.
