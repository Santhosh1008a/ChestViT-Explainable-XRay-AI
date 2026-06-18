"""
data/dataset.py
---------------
PyTorch Dataset for NIH ChestX-ray14.

Key design decisions:
  - Uses Hugging Face `datasets` for streaming from `BahaaEldin0/NIH-Chest-Xray-14`.
  - Avoids 45GB local download, scales natively up to 112k images.
  - Multi-hot label encoding for 14 simultaneous disease labels.
  - CLAHE preprocessing baked into the transform pipeline.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
import numpy as np
import logging
import cv2
from PIL import Image

from data.preprocessing import get_train_transforms, get_val_transforms, apply_clahe

logger = logging.getLogger(__name__)

# ── 14 NIH disease labels ─────────────────────────────────────────────────────
DISEASE_LABELS = [
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

NUM_CLASSES = len(DISEASE_LABELS)

def encode_labels(labels) -> np.ndarray:
    """
    Convert a list of string labels (or a pipe separated string) into a 14-dim multi-hot vector.
    """
    if isinstance(labels, str):
        if labels == "No Finding":
            labels_list = []
        else:
            labels_list = labels.split("|")
    else:
        labels_list = labels

    label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    for disease in labels_list:
        disease = disease.strip()
        if disease in DISEASE_LABELS:
            label_vec[DISEASE_LABELS.index(disease)] = 1.0
    return label_vec

class HFStreamingChestXrayDataset(IterableDataset):
    """
    Wrapper around Hugging Face IterableDataset for PyTorch DataLoader compatibility.
    """
    def __init__(
        self,
        hf_iterable,
        image_size: int = 224,
        transform=None,
        clip_limit: float = 2.0,
        tile_size: int = 8,
        take_limit: int = None
    ):
        self.hf_iterable = hf_iterable
        self.image_size = image_size
        self.transform = transform
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.take_limit = take_limit

    def __iter__(self):
        iterable = self.hf_iterable
        if self.take_limit is not None:
            iterable = iterable.take(self.take_limit)

        for idx, item in enumerate(iterable):
            image = item.get('image') or item.get('Image')
            if image is None:
                continue

            # Ensure image is PIL Image and convert to grayscale numpy array
            if not isinstance(image, Image.Image):
                continue

            img_gray = np.array(image.convert("L"))

            # CLAHE
            img_clahe = apply_clahe(img_gray, clip_limit=self.clip_limit, tile_size=self.tile_size)

            # Resize BEFORE stacking to save memory/computation
            img_resized = cv2.resize(img_clahe, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

            # Convert to RGB numpy array for Albumentations
            img_rgb = np.stack([img_resized, img_resized, img_resized], axis=-1)

            # Transform
            if self.transform:
                augmented = self.transform(image=img_rgb)
                img_tensor = augmented["image"]
            else:
                img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0

            # Labels
            label_str_list = item.get('label', [])
            label_vec = encode_labels(label_str_list)
            label_tensor = torch.tensor(label_vec, dtype=torch.float32)

            # Yield path as Patient ID or fallback to str index to satisfy interface
            patient_id = str(item.get('Patient ID', idx))

            yield img_tensor, label_tensor, patient_id

def build_dataloaders(
    images_dir=None,
    labels_csv=None,
    train_list_txt=None,
    test_list_txt=None,
    image_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
    val_split: float = 0.1,
    train_fraction: float = 1.0,
    seed: int = 42,
    train_take_limit: int = 20000,
    val_take_limit: int = 2000,
):
    """
    Build train, validation, and test DataLoaders directly from the HF Stream.
    Uses generic fallback args for compatibility with old code calls.
    """
    print(f"  [Dataset] Loading HF Stream BahaaEldin0/NIH-Chest-Xray-14...")

    # In order to stream properly we load each split
    ds = load_dataset('BahaaEldin0/NIH-Chest-Xray-14', streaming=True)

    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    # Note: BahaaEldin0 dataset has 'train', 'valid', 'test' splits
    train_stream = ds['train']
    val_stream = ds['valid']
    test_stream = ds['test']

    # Shuffle the training stream slightly for better randomness (buffer_size=1000)
    train_stream = train_stream.shuffle(buffer_size=1000, seed=seed)

    # If train_fraction < 1.0, we can adjust take limits accordingly.
    if train_fraction < 1.0 and train_take_limit is not None:
        train_take_limit = int(train_take_limit * train_fraction)

    train_ds = HFStreamingChestXrayDataset(train_stream, image_size=image_size, transform=train_transform, take_limit=train_take_limit)
    val_ds = HFStreamingChestXrayDataset(val_stream, image_size=image_size, transform=val_transform, take_limit=val_take_limit)
    test_ds = HFStreamingChestXrayDataset(test_stream, image_size=image_size, transform=val_transform, take_limit=val_take_limit)

    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_loader = DataLoader(train_ds, **dl_kwargs)
    val_loader = DataLoader(val_ds, **dl_kwargs)
    test_loader = DataLoader(test_ds, **dl_kwargs)

    # Calculate uniform weights (streaming mode makes exact calculation slow)
    class_weights = torch.ones(NUM_CLASSES, dtype=torch.float32)

    print(f"\n  DataLoaders ready:")
    print(f"    Train limit: {train_take_limit if train_take_limit else 'Full'}")
    print(f"    Val/Test limit: {val_take_limit if val_take_limit else 'Full'}")

    return train_loader, val_loader, test_loader, class_weights
