"""
models/vit_model.py
-------------------
ViT-Base-16 fine-tuned for multi-label chest X-ray classification.

Architecture:
  - Backbone: google/vit-base-patch16-224-in21k (pre-trained on ImageNet-21k)
  - Head: Linear(768 → 14) — replaces the [CLS] token classifier
  - Activation: Sigmoid (multi-label, not softmax)
  - Attention: output_attentions=True exposes all 12 transformer layer
    attention weight tensors for Attention Rollout visualization

RTX 3050 Optimizations:
  - gradient_checkpointing: reduces VRAM by ~30% by recomputing activations
    during backprop instead of caching them
  - Mixed precision (fp16) used during training (handled in train.py)

Reference:
  Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
  https://arxiv.org/abs/2010.11929
"""

from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class ChestViT(nn.Module):
    """
    ViT-Base-16 with a multi-label classification head.

    The model exposes two outputs:
      1. logits  — shape (B, 14), raw pre-sigmoid scores
      2. attentions — list of 12 tensors, each (B, num_heads, seq_len, seq_len)
         seq_len = 197 = 1 (CLS) + 196 (14×14 patches)
         Only returned when output_attentions=True (set during initialization).

    Usage:
        model = ChestViT(num_classes=14)
        logits, attentions = model(pixel_values, output_attentions=True)
        probs = torch.sigmoid(logits)
    """

    def __init__(
        self,
        num_classes: int = 14,
        pretrained_name: str = "google/vit-base-patch16-224-in21k",
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ── Load pre-trained ViT backbone ─────────────────────────────────────
        print(f"  Loading ViT backbone: {pretrained_name}")
        self.vit = ViTModel.from_pretrained(
            pretrained_name,
            add_pooling_layer=False,   # We extract [CLS] ourselves
            output_attentions=True,    # Always expose attention weights
        )

        # ── RTX 3050: gradient checkpointing ─────────────────────────────────
        if gradient_checkpointing:
            self.vit.gradient_checkpointing_enable()
            print("  Gradient checkpointing: ENABLED (saves ~30% VRAM)")

        # ── Multi-label classification head ───────────────────────────────────
        hidden_size = self.vit.config.hidden_size  # 768 for ViT-Base
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Initialize head with small weights (better multi-label convergence)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        print(f"  Classification head: Linear({hidden_size} → {num_classes})")
        print(f"  Total parameters: {self.count_parameters():,.0f}")
        print(f"  Trainable parameters: {self.count_parameters(trainable_only=True):,.0f}")

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_attentions: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass.

        Args:
            pixel_values:      (B, 3, 224, 224) normalized image tensor.
            output_attentions: Return attention weights for rollout visualization.

        Returns:
            logits:     (B, 14) raw pre-sigmoid classification scores.
            attentions: List of 12 tensors (B, 12, 197, 197), or None.
        """
        outputs = self.vit(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
        )

        # [CLS] token representation — shape: (B, 768)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        # Multi-label logits — shape: (B, 14)
        logits = self.classifier(cls_output)

        # Attention weights: tuple of 12 tensors, each (B, 12, 197, 197)
        attentions = outputs.attentions if output_attentions else None

        return logits, attentions

    def count_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_patch_size(self) -> int:
        """Return the patch size (16 for ViT-Base-16)."""
        return self.vit.config.patch_size  # 16

    def get_num_patches(self) -> int:
        """Return number of patches per side (14 for 224/16)."""
        img_size = self.vit.config.image_size  # 224
        patch_size = self.vit.config.patch_size  # 16
        return img_size // patch_size  # 14


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    num_classes: int = 14,
) -> ChestViT:
    """
    Load a saved ChestViT checkpoint.

    Args:
        checkpoint_path: Path to .pt or .pth checkpoint file.
        device:          Target device (cuda / cpu).
        num_classes:     Must match the saved model.

    Returns:
        Loaded ChestViT model in eval mode.
    """
    checkpoint_path = Path(checkpoint_path)
    print(f"  Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ChestViT(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  Checkpoint loaded from epoch {checkpoint.get('epoch', '?')} "
          f"(val_auc={checkpoint.get('val_auc', '?'):.4f})")
    return model


def save_checkpoint(
    model: ChestViT,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_auc: float,
    save_path: str | Path,
) -> None:
    """Save a training checkpoint."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "val_auc": val_auc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, save_path)
    print(f"  Checkpoint saved → {save_path} (val_auc={val_auc:.4f})")
