from __future__ import annotations

from pathlib import Path

from torchvision import datasets, transforms


IMAGE_SIZE = 64


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((72, 72)),
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.15),
                transforms.RandomRotation(degrees=8),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_imagefolder(root: str | Path, train: bool) -> datasets.ImageFolder:
    return datasets.ImageFolder(str(root), transform=build_transforms(train=train))
