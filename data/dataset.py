"""
data/dataset.py
---------------
PyTorch Dataset for NIH ChestX-ray14.

Key design decisions:
  - Patient-level split via official train_val_list.txt / test_list.txt
    (prevents data leakage across the 112K images from 30K patients)
  - Multi-hot label encoding for 14 simultaneous disease labels
  - CLAHE preprocessing baked into __getitem__ (cached via raw pre-processing)
  - Returns (image_tensor, label_vector, image_path) for explainability use
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from data.preprocessing import load_and_preprocess_raw, get_train_transforms, get_val_transforms


# ── 14 NIH disease labels ─────────────────────────────────────────────────────
DISEASE_LABELS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

NUM_CLASSES = len(DISEASE_LABELS)  # 14


def encode_labels(finding_str: str) -> np.ndarray:
    """
    Convert NIH 'Finding Labels' string to a 14-dim multi-hot vector.

    Example:
        "Atelectasis|Effusion" → [1, 0, 1, 0, ..., 0]
        "No Finding"           → [0, 0, 0, 0, ..., 0]

    Args:
        finding_str: Pipe-delimited string from Data_Entry_2017.csv.

    Returns:
        np.ndarray float32 of shape (14,).
    """
    label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    if finding_str == "No Finding":
        return label_vec
    for disease in finding_str.split("|"):
        disease = disease.strip()
        if disease in DISEASE_LABELS:
            label_vec[DISEASE_LABELS.index(disease)] = 1.0
    return label_vec


class ChestXrayDataset(Dataset):
    """
    NIH ChestX-ray14 Dataset.

    Args:
        images_dir:     Directory containing all .png X-ray files.
        labels_csv:     Path to Data_Entry_2017.csv.
        image_list:     List of image filenames (e.g. from train_val_list.txt).
        transform:      Albumentations transform to apply after CLAHE.
        train_fraction: Subsample fraction (for quick RTX 3050 experiments).
        seed:           Random seed for subsampling reproducibility.
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_csv: str | Path,
        image_list: List[str],
        transform=None,
        train_fraction: float = 1.0,
        seed: int = 42,
    ):
        self.images_dir = Path(images_dir)
        self.transform = transform

        # Load metadata
        df = pd.read_csv(labels_csv)
        df = df.set_index("Image Index")

        # Filter to the provided image list and only keep files that exist
        valid = [
            fname for fname in image_list
            if (self.images_dir / fname).exists()
        ]
        if len(valid) < len(image_list):
            missing = len(image_list) - len(valid)
            print(f"  [Dataset] Warning: {missing} images not found on disk, skipping.")

        # Subsample for quick iteration
        if train_fraction < 1.0:
            rng = random.Random(seed)
            valid = rng.sample(valid, max(1, int(len(valid) * train_fraction)))

        self.image_names: List[str] = valid
        self.labels_df = df

        # Pre-compute label matrix for efficiency
        self.labels = np.stack([
            encode_labels(df.loc[fname, "Finding Labels"])
            if fname in df.index else np.zeros(NUM_CLASSES, dtype=np.float32)
            for fname in self.image_names
        ])  # shape: (N, 14)

        print(f"  [Dataset] Loaded {len(self.image_names):,} images, "
              f"{NUM_CLASSES} classes, "
              f"positive rate: {self.labels.mean(axis=0).mean():.3f}")

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            image_tensor: (3, H, W) float32 tensor, normalized.
            label_tensor: (14,) float32 multi-hot vector.
            image_path:   Absolute path string (for explainability callbacks).
        """
        fname = self.image_names[idx]
        img_path = self.images_dir / fname

        # CLAHE preprocessing (returns HWC uint8)
        image = load_and_preprocess_raw(str(img_path))

        # Albumentations transform (normalize + optional augmentation)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # torch.Tensor (C, H, W)
        else:
            # Fallback: just convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label, str(img_path)

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute per-class positive weights for weighted BCE loss.

        Formula: weight_c = (N - n_pos_c) / n_pos_c
        (Standard approach for severe class imbalance in medical imaging.)

        Returns:
            Tensor of shape (14,) — higher weight = rarer disease.
        """
        n = len(self.image_names)
        n_pos = self.labels.sum(axis=0)  # (14,)
        n_pos = np.clip(n_pos, 1, None)  # avoid division by zero
        weights = (n - n_pos) / n_pos
        # Cap at 20 to prevent training instability for very rare diseases
        weights = np.clip(weights, 1.0, 20.0)
        return torch.tensor(weights, dtype=torch.float32)


def read_image_list(txt_path: str | Path) -> List[str]:
    """Read a line-separated list of image filenames from a .txt file."""
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def build_dataloaders(
    images_dir: str | Path,
    labels_csv: str | Path,
    train_list_txt: str | Path,
    test_list_txt: str | Path,
    image_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
    val_split: float = 0.1,
    train_fraction: float = 1.0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Build train, validation, and test DataLoaders.

    Splits the official train_val_list.txt into train/val using val_split ratio
    (patient-level, not random image-level — preserving the official test set).

    Returns:
        (train_loader, val_loader, test_loader, class_weights)
    """
    all_train_val = read_image_list(train_list_txt)
    test_list = read_image_list(test_list_txt)

    # Patient-level val split from train_val
    rng = random.Random(seed)
    rng.shuffle(all_train_val)
    n_val = max(1, int(len(all_train_val) * val_split))
    val_list   = all_train_val[:n_val]
    train_list = all_train_val[n_val:]

    train_transform = get_train_transforms(image_size)
    val_transform   = get_val_transforms(image_size)

    train_ds = ChestXrayDataset(
        images_dir, labels_csv, train_list,
        transform=train_transform,
        train_fraction=train_fraction,
        seed=seed,
    )
    val_ds = ChestXrayDataset(
        images_dir, labels_csv, val_list,
        transform=val_transform,
        train_fraction=1.0,  # Always use full val set
        seed=seed,
    )
    test_ds = ChestXrayDataset(
        images_dir, labels_csv, test_list,
        transform=val_transform,
        train_fraction=1.0,
        seed=seed,
    )

    dl_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **dl_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False, **dl_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False, **dl_kwargs
    )

    class_weights = train_ds.get_class_weights()

    print(f"\n  DataLoaders ready:")
    print(f"    Train  : {len(train_ds):>7,} images | {len(train_loader):>5,} batches")
    print(f"    Val    : {len(val_ds):>7,} images | {len(val_loader):>5,} batches")
    print(f"    Test   : {len(test_ds):>7,} images | {len(test_loader):>5,} batches")
    print(f"    Class weights (top 3): "
          f"{dict(zip(DISEASE_LABELS, class_weights.tolist()))}")

    return train_loader, val_loader, test_loader, class_weights
