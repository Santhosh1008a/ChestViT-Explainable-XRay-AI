"""
data/preprocessing.py
---------------------
Image preprocessing pipeline for NIH ChestX-ray14.

Strategy:
  1. CLAHE (Contrast Limited Adaptive Histogram Equalization) — boosts subtle
     lung texture visibility in low-contrast X-rays.
  2. Albumentations augmentation pipeline — radiologically realistic transforms.
  3. ViT normalization with ImageNet-21k stats (the pre-training distribution).

CLAHE rationale:
  Chest X-rays have a very narrow dynamic range. CLAHE locally equalizes
  contrast in small tile windows, revealing subtle nodules and infiltrates
  that global equalization would miss.
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from typing import Tuple


# ── ImageNet-21k normalization (matches ViT pre-training) ─────────────────────
IMAGENET_MEAN = (0.5, 0.5, 0.5)
IMAGENET_STD  = (0.5, 0.5, 0.5)
# Note: google/vit-base-patch16-224-in21k uses [-1, 1] normalization (mean=0.5, std=0.5)


def apply_clahe(image_gray: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE to a grayscale X-ray image.

    Args:
        image_gray: Single-channel uint8 grayscale image (H, W).
        clip_limit: Threshold for contrast limiting. Higher = more contrast boost.
        tile_size:  Grid size for local histogram equalization.

    Returns:
        CLAHE-enhanced grayscale image (H, W) as uint8.
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )
    return clahe.apply(image_gray)


def load_and_preprocess_raw(
    image_path: str,
    output_size: Tuple[int, int] = (224, 224),
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """
    Load a chest X-ray PNG, apply CLAHE, and convert to 3-channel RGB.

    The NIH images are grayscale PNGs saved as 8-bit or 16-bit.
    We:
      1. Read as grayscale
      2. Normalize 16-bit → 8-bit if needed
      3. Apply CLAHE
      4. Resize to output_size
      5. Convert to 3-channel (replicate gray → R, G, B)

    Args:
        image_path:  Path to the .png X-ray file.
        output_size: (width, height) tuple for resizing.
        clip_limit:  CLAHE clip limit.
        tile_size:   CLAHE tile grid size.

    Returns:
        np.ndarray of shape (H, W, 3) uint8 — ready for Albumentations.
    """
    # Read as grayscale (handles both 8-bit and 16-bit)
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Normalize 16-bit to 8-bit
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)

    # Apply CLAHE
    img = apply_clahe(img, clip_limit=clip_limit, tile_size=tile_size)

    # Resize
    img = cv2.resize(img, output_size, interpolation=cv2.INTER_AREA)

    # Convert grayscale → 3-channel RGB (ViT expects 3 channels)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img  # shape: (H, W, 3), dtype: uint8


# ── Albumentations Pipelines ─────────────────────────────────────────────────

def get_train_transforms(image_size: int = 224) -> A.Compose:
    """
    Training augmentation pipeline.

    Augmentations chosen for radiological realism:
      - HorizontalFlip: Patients can be imaged in either orientation
      - ShiftScaleRotate: Minor positioning variance (±10°, scale ±10%)
      - RandomBrightnessContrast: Exposure variation between X-ray machines
      - GridDistortion: Simulates slight body movement artifacts
      - GaussNoise: Detector noise

    No vertical flip — lungs have a fixed anatomical orientation.
    No heavy color jitter — X-rays are monochrome.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.90, 1.10),
            rotate=(-10, 10),
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.4,
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.05,
            p=0.2,
        ),
        A.GaussNoise(std_range=(0.02, 0.10), p=0.2),
        # Normalize using ImageNet-21k stats
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),  # (H, W, C) → (C, H, W), float32
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """
    Validation / inference transforms — no augmentation, only normalize + tensorize.
    """
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def denormalize(tensor_image) -> np.ndarray:
    """
    Reverse ImageNet normalization for visualization.

    Args:
        tensor_image: torch.Tensor (C, H, W) in normalized space.

    Returns:
        np.ndarray (H, W, 3) uint8 for display.
    """
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor_image.cpu().float() * std + mean  # denormalize
    img = img.permute(1, 2, 0).numpy()             # (C, H, W) → (H, W, C)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img
