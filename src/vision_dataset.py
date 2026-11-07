"""
src/vision_dataset.py
PyTorch Dataset and DataLoader definitions for:
1. Enhanced Indian Food (Primary 43-Class Benchmark)
2. FoodLens (Secondary / Auxiliary Benchmark)
"""

import json
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CACHE_DIR = Path("data/processed/images/enhanced_cache")
SPLITS_DIR = Path("data/processed/vision_splits")
MODELS_DIR = Path("models")

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_vision_transforms(split: str = "train", img_size: int = 224) -> transforms.Compose:
    """
    Returns data transforms for train, validation, or test splits.
    
    Training:
      - RandomResizedCrop(224, scale=(0.8, 1.0))
      - RandomHorizontalFlip(p=0.5)
      - RandomRotation(degrees=15)
      - ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
      - ToTensor()
      - Normalize(ImageNet mean & std)
      
    Val / Test:
      - Resize(256)
      - CenterCrop(224)
      - ToTensor()
      - Normalize(ImageNet mean & std)
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class EnhancedFoodDataset(Dataset):
    """
    PyTorch Dataset for Enhanced Indian Food 43-class dataset.
    Loads images directly from on-disk cache using MD5 hashes.
    """
    def __init__(
        self,
        split_source: Union[str, Path, pd.DataFrame],
        cache_dir: Union[str, Path] = CACHE_DIR,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[dict] = None
    ):
        if isinstance(split_source, (str, Path)):
            self.df = pd.read_csv(split_source)
        elif isinstance(split_source, pd.DataFrame):
            self.df = split_source.copy()
        else:
            raise TypeError("split_source must be a path or a DataFrame")

        self.cache_dir = Path(cache_dir)
        self.transform = transform

        if class_to_idx is None:
            class_to_idx_path = MODELS_DIR / "class_to_idx.json"
            if class_to_idx_path.exists():
                with open(class_to_idx_path, "r", encoding="utf-8") as f:
                    self.class_to_idx = json.load(f)
            else:
                classes = sorted(self.df["class_name"].unique())
                self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Verify class_idx in dataframe
        if "class_idx" not in self.df.columns:
            self.df["class_idx"] = self.df["class_name"].map(self.class_to_idx)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        row = self.df.iloc[idx]
        h = row["md5_hash"]
        img_path = self.cache_dir / f"{h}.jpg"

        if not img_path.exists():
            raise FileNotFoundError(f"Image hash {h} not found in {self.cache_dir}")

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        target = int(row["class_idx"])
        class_name = str(row["class_name"])
        return img, target, class_name, h


class FoodLensDataset(Dataset):
    """
    PyTorch Dataset for Auxiliary FoodLens benchmark.
    Loads images from individual raw paths on disk.
    """
    def __init__(
        self,
        split_source: Union[str, Path, pd.DataFrame],
        transform: Optional[Callable] = None,
        class_to_idx: Optional[dict] = None
    ):
        if isinstance(split_source, (str, Path)):
            self.df = pd.read_csv(split_source)
        elif isinstance(split_source, pd.DataFrame):
            self.df = split_source.copy()
        else:
            raise TypeError("split_source must be a path or a DataFrame")

        self.transform = transform

        if class_to_idx is None:
            classes = sorted(self.df["class_name"].unique())
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])

        if not img_path.exists():
            raise FileNotFoundError(f"FoodLens image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        class_name = str(row["class_name"])
        target = self.class_to_idx.get(class_name, -1)
        h = str(row.get("md5_hash", ""))
        return img, target, class_name, h


def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, val, and test splits.
    """
    train_tf = get_vision_transforms("train", img_size=img_size)
    eval_tf = get_vision_transforms("val", img_size=img_size)

    train_ds = EnhancedFoodDataset(SPLITS_DIR / "train_split.csv", transform=train_tf)
    val_ds = EnhancedFoodDataset(SPLITS_DIR / "val_split.csv", transform=eval_tf)
    test_ds = EnhancedFoodDataset(SPLITS_DIR / "test_split.csv", transform=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    return train_loader, val_loader, test_loader
