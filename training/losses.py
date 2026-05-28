"""
training/losses.py
------------------
Loss functions for multi-label chest X-ray classification.

NIH ChestX-ray14 has severe class imbalance:
  - "No Finding" accounts for ~53% of records
  - Hernia is present in <0.2% of images
  - Most disease classes are 1–10% positive rate

Standard BCE would be dominated by negative examples.
We use BCEWithLogitsLoss with pos_weight to up-weight rare disease predictions.

pos_weight[c] = (N_neg_c / N_pos_c) — per PyTorch docs recommendation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Weighted Binary Cross-Entropy with Logits loss for multi-label classification.

    This is the standard loss for medical imaging multi-label tasks.
    See NIH ChestX-ray14 paper (Wang et al., 2017) and subsequent work.

    Args:
        pos_weight: Tensor of shape (num_classes,) with per-class positive weights.
                    Computed from training set statistics.
                    Higher weight → penalizes missing positive predictions more.
        reduction:  "mean" (default) or "sum".
    """

    def __init__(
        self,
        pos_weight: torch.Tensor,
        reduction: str = "mean",
    ):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, 14) — raw pre-sigmoid model outputs.
            targets: (B, 14) — float32 multi-hot label vectors.

        Returns:
            Scalar loss value.
        """
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification — alternative to weighted BCE.

    Focal loss further down-weights easy negatives, focusing training on
    hard examples (borderline cases). Can help with extreme imbalance.

    Lin et al., 2017: https://arxiv.org/abs/1708.02002

    Args:
        gamma:      Focusing parameter (0 = standard BCE). Typical: 1.0–2.0.
        pos_weight: Optional per-class positive weights.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        # Focal term: (1 - pt)^gamma
        pt = torch.where(targets == 1, p, 1 - p)
        focal_weight = (1 - pt) ** self.gamma
        loss = (focal_weight * bce).mean()
        return loss


def build_loss(
    loss_type: str,
    class_weights: torch.Tensor,
    device: torch.device,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """
    Factory function to build the training loss.

    Args:
        loss_type:     "weighted_bce" (default) or "focal".
        class_weights: (14,) tensor from dataset.get_class_weights().
        device:        Target device.
        focal_gamma:   Gamma for focal loss (only used if loss_type="focal").

    Returns:
        Configured loss module.
    """
    class_weights = class_weights.to(device)
    if loss_type == "weighted_bce":
        print(f"  Loss: WeightedBCEWithLogitsLoss")
        return WeightedBCEWithLogitsLoss(pos_weight=class_weights)
    elif loss_type == "focal":
        print(f"  Loss: FocalLoss(gamma={focal_gamma})")
        return FocalLoss(gamma=focal_gamma, pos_weight=class_weights)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'weighted_bce' or 'focal'.")
