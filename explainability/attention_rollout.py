"""
explainability/attention_rollout.py
------------------------------------
ViT Attention Rollout — the definitive ViT explainability technique.

Algorithm (Abnar & Zuidema, 2020 — "Quantifying Attention Flow in Transformers"):
  1. For each of 12 transformer layers, average attention weights across all
     12 heads → shape per layer: (seq_len, seq_len) = (197, 197)
  2. Add the identity matrix (residual connections preserve information)
  3. Re-normalize each row to sum to 1
  4. Multiply all 12 matrices together in sequence (matrix chain multiplication)
     → This propagates attention from input patches all the way to [CLS]
  5. Extract the [CLS] row → shape: (197,) = 1 CLS + 196 patches
  6. Reshape the 196 patch values to (14, 14) — the spatial patch grid
  7. Upsample to (224, 224) — the original image resolution
  8. Overlay on the original image as a heatmap

Why Attention Rollout > Grad-CAM for ViT:
  - Grad-CAM was designed for CNNs with spatial feature maps
  - ViT has no intermediate spatial feature maps — Grad-CAM produces
    blurry, uninformative results on pure transformers
  - Attention Rollout correctly accounts for skip connections and is
    mathematically derived from the transformer's own attention flow

Reference:
  Abnar, S. & Zuidema, W. (2020). Quantifying Attention Flow in Transformers.
  arXiv:2005.00928
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def compute_attention_rollout(
    attentions: List[torch.Tensor],
    head_fusion: str = "mean",
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """
    Compute Attention Rollout from a list of per-layer attention tensors.

    Args:
        attentions:     List of 12 tensors, each shape (B, H, N, N).
                        B = batch, H = 12 heads, N = 197 tokens (1 CLS + 196 patches).
        head_fusion:    How to combine attention heads. Options:
                        "mean"  — average across heads (standard rollout)
                        "max"   — take max across heads (more focused)
                        "min"   — take min across heads (conservative)
        discard_ratio:  Fraction of lowest-attention patch weights to zero out.
                        Helps focus the heatmap on the most attended patches.
                        Set to 0.0 to disable.

    Returns:
        np.ndarray of shape (14, 14) — attention map over the 14×14 patch grid.
        Values are normalized to [0, 1].
    """
    # ── 1. Process each attention layer ───────────────────────────────────────
    # Start with identity (represents perfect self-attention at layer 0)
    batch_size = attentions[0].shape[0]
    num_tokens = attentions[0].shape[-1]  # 197

    result = torch.eye(num_tokens, device=attentions[0].device)
    result = result.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 197, 197)

    for attention in attentions:
        # attention: (B, H, N, N)
        # Fuse heads
        if head_fusion == "mean":
            attention_fused = attention.mean(dim=1)          # (B, N, N)
        elif head_fusion == "max":
            attention_fused = attention.max(dim=1).values    # (B, N, N)
        elif head_fusion == "min":
            attention_fused = attention.min(dim=1).values    # (B, N, N)
        else:
            raise ValueError(f"Unknown head_fusion: {head_fusion}")

        # Discard low-attention tokens (remove noise)
        if discard_ratio > 0.0:
            flat = attention_fused.view(batch_size, -1)
            threshold_idx = int(flat.shape[-1] * discard_ratio)
            # Find the threshold value
            sorted_flat, _ = flat.sort(dim=-1)
            threshold = sorted_flat[:, threshold_idx].unsqueeze(-1).unsqueeze(-1)
            attention_fused = torch.where(
                attention_fused > threshold,
                attention_fused,
                torch.zeros_like(attention_fused),
            )

        # Add residual connection (identity skip)
        attention_fused = attention_fused + torch.eye(
            num_tokens, device=attention_fused.device
        ).unsqueeze(0)

        # Row-normalize (each token's attention distribution sums to 1)
        row_sums = attention_fused.sum(dim=-1, keepdim=True)
        attention_fused = attention_fused / (row_sums + 1e-8)

        # Chain-multiply: propagate attention through layers
        result = torch.matmul(attention_fused, result)  # (B, N, N)

    # ── 2. Extract [CLS] → patch attention ────────────────────────────────────
    # CLS token is index 0; its row shows which patches it attends to
    cls_attn = result[:, 0, 1:]  # (B, 196) — skip the CLS self-attention

    # ── 3. Reshape to patch grid ───────────────────────────────────────────────
    patch_grid_size = int(cls_attn.shape[-1] ** 0.5)  # 14
    cls_attn = cls_attn.reshape(batch_size, patch_grid_size, patch_grid_size)  # (B, 14, 14)

    # Normalize to [0, 1]
    for b in range(batch_size):
        v_min = cls_attn[b].min()
        v_max = cls_attn[b].max()
        cls_attn[b] = (cls_attn[b] - v_min) / (v_max - v_min + 1e-8)

    # Return as numpy — usually called with batch_size=1 for visualization
    return cls_attn.detach().cpu().numpy()  # (B, 14, 14)


def rollout_to_heatmap(
    rollout_map: np.ndarray,
    original_image: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.5,
    image_size: int = 224,
) -> np.ndarray:
    """
    Upsample the 14×14 rollout map and overlay it on the original image.

    Args:
        rollout_map:    (14, 14) or (1, 14, 14) float32 array in [0, 1].
        original_image: (H, W, 3) uint8 RGB image (before normalization).
        colormap:       OpenCV colormap constant (default: COLORMAP_JET).
        alpha:          Heatmap overlay opacity (0=transparent, 1=opaque).
        image_size:     Target size for both map and image (default: 224).

    Returns:
        np.ndarray (H, W, 3) uint8 — heatmap overlay on original image.
    """
    if rollout_map.ndim == 3:
        rollout_map = rollout_map[0]  # (14, 14)

    # Upsample 14×14 → 224×224
    heatmap = cv2.resize(rollout_map, (image_size, image_size),
                         interpolation=cv2.INTER_CUBIC)

    # Normalize to [0, 255] for colormap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Apply colormap → (H, W, 3) BGR
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

    # Convert to RGB for matplotlib/Gradio
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Ensure original image is 224×224 RGB
    if original_image.shape[:2] != (image_size, image_size):
        original_image = cv2.resize(original_image, (image_size, image_size))
    if original_image.ndim == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    # Alpha blend
    overlay = cv2.addWeighted(
        original_image.astype(np.float32), 1 - alpha,
        heatmap_colored_rgb.astype(np.float32), alpha,
        0,
    ).astype(np.uint8)

    return overlay


def visualize_rollout(
    rollout_map: np.ndarray,
    original_image: np.ndarray,
    title: str = "Attention Rollout",
    disease_scores: Optional[np.ndarray] = None,
    disease_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a rich matplotlib figure with:
      - Original CLAHE-enhanced X-ray
      - Attention rollout heatmap
      - Overlay (blended)
      - Optional disease probability bar chart

    Args:
        rollout_map:    (14, 14) attention rollout array.
        original_image: (H, W, 3) uint8 RGB original image.
        title:          Figure title.
        disease_scores: Optional (14,) array of sigmoid probabilities.
        disease_names:  Optional list of 14 disease name strings.
        save_path:      If given, saves the figure to this path.

    Returns:
        matplotlib Figure object.
    """
    overlay = rollout_to_heatmap(rollout_map, original_image)

    if disease_scores is not None and disease_names is not None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        n_panels = 4
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        n_panels = 3

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")

    # Panel 1: Original image
    axes[0].imshow(original_image, cmap="gray" if original_image.ndim == 2 else None)
    axes[0].set_title("Original X-Ray (CLAHE)", color="white", fontsize=11)
    axes[0].axis("off")

    # Panel 2: Raw rollout map (14×14 upsampled)
    rollout_upsampled = cv2.resize(rollout_map if rollout_map.ndim == 2 else rollout_map[0],
                                   (224, 224), interpolation=cv2.INTER_CUBIC)
    im = axes[1].imshow(rollout_upsampled, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Attention Rollout Map", color="white", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Heatmap Overlay", color="white", fontsize=11)
    axes[2].axis("off")

    # Panel 4 (optional): Disease scores bar chart
    if disease_scores is not None and disease_names is not None and n_panels == 4:
        sorted_idx = np.argsort(disease_scores)[::-1]
        colors = [
            "#ef4444" if disease_scores[i] > 0.5 else
            "#f97316" if disease_scores[i] > 0.3 else "#3b82f6"
            for i in sorted_idx
        ]
        bars = axes[3].barh(
            [disease_names[i] for i in sorted_idx],
            [disease_scores[i] for i in sorted_idx],
            color=colors,
        )
        axes[3].set_xlim(0, 1)
        axes[3].axvline(x=0.5, color="white", linestyle="--", alpha=0.5, label="Threshold")
        axes[3].set_xlabel("Probability", color="white")
        axes[3].set_title("Disease Predictions", color="white", fontsize=11)
        axes[3].tick_params(colors="white")
        axes[3].spines["bottom"].set_color("white")
        axes[3].spines["left"].set_color("white")
        axes[3].spines["top"].set_visible(False)
        axes[3].spines["right"].set_visible(False)
        for spine in axes[3].spines.values():
            spine.set_edgecolor("white")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Visualization saved → {save_path}")

    return fig


@torch.no_grad()
def explain_prediction(
    model,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    device: torch.device,
    disease_names: Optional[List[str]] = None,
    head_fusion: str = "mean",
    discard_ratio: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full inference + explainability pipeline for a single image.

    Args:
        model:          ChestViT model in eval mode.
        image_tensor:   (1, 3, 224, 224) normalized tensor.
        original_image: (H, W, 3) uint8 original image for overlay.
        device:         torch.device.
        disease_names:  List of 14 disease names.
        head_fusion:    Attention head fusion strategy.
        discard_ratio:  Low-attention token discard ratio.

    Returns:
        probs:      (14,) numpy array of sigmoid probabilities.
        rollout:    (14, 14) numpy array — attention rollout map.
        overlay:    (224, 224, 3) uint8 numpy array — heatmap overlay.
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    logits, attentions = model(image_tensor, output_attentions=True)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # (14,)

    rollout = compute_attention_rollout(
        attentions, head_fusion=head_fusion, discard_ratio=discard_ratio
    )  # (1, 14, 14)

    overlay = rollout_to_heatmap(rollout[0], original_image)  # (224, 224, 3)

    return probs, rollout[0], overlay
