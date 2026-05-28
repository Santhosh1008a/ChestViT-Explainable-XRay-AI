"""
tests/test_modules.py
----------------------
Unit tests for the core pipeline modules.

Run with:
    python -m pytest tests/ -v

Tests are designed to run WITHOUT the NIH dataset (uses synthetic data).
GPU not required — tests run on CPU.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import pytest

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_clahe_output_shape(self):
        """CLAHE should preserve image dimensions."""
        from data.preprocessing import apply_clahe
        gray = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        result = apply_clahe(gray)
        assert result.shape == (512, 512), f"Expected (512,512), got {result.shape}"
        assert result.dtype == np.uint8

    def test_clahe_output_range(self):
        """CLAHE output should remain in [0, 255]."""
        from data.preprocessing import apply_clahe
        gray = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        result = apply_clahe(gray)
        assert result.min() >= 0 and result.max() <= 255

    def test_train_transform_output_type(self):
        """Training transform should return a float32 tensor."""
        from data.preprocessing import get_train_transforms
        transform = get_train_transforms(image_size=224)
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = transform(image=dummy_image)["image"]
        assert isinstance(result, torch.Tensor), "Expected torch.Tensor"
        assert result.dtype == torch.float32
        assert result.shape == (3, 224, 224), f"Expected (3,224,224), got {result.shape}"

    def test_val_transform_deterministic(self):
        """Val transform should be deterministic (no random augmentation)."""
        from data.preprocessing import get_val_transforms
        transform = get_val_transforms(224)
        dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        t1 = transform(image=dummy.copy())["image"]
        t2 = transform(image=dummy.copy())["image"]
        assert torch.allclose(t1, t2), "Val transform should be deterministic"

    def test_denormalize_range(self):
        """Denormalized image should be in [0, 255] uint8 range."""
        from data.preprocessing import denormalize
        # Simulate normalized tensor
        tensor = torch.randn(3, 224, 224) * 0.5  # ~within [-1, 1] after normalization
        result = denormalize(tensor)
        assert result.dtype == np.uint8
        assert result.shape == (224, 224, 3)
        assert result.min() >= 0 and result.max() <= 255


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataset:
    def test_label_encoding_no_finding(self):
        """'No Finding' should encode to all-zeros."""
        from data.dataset import encode_labels
        result = encode_labels("No Finding")
        assert result.shape == (14,)
        assert result.sum() == 0.0, "No Finding should have no positive labels"

    def test_label_encoding_single_disease(self):
        """Single disease should have exactly one positive label."""
        from data.dataset import encode_labels, DISEASE_LABELS
        for i, disease in enumerate(DISEASE_LABELS):
            result = encode_labels(disease)
            assert result[i] == 1.0, f"{disease} should be positive at index {i}"
            assert result.sum() == 1.0, f"Only one positive label for {disease}"

    def test_label_encoding_multi_disease(self):
        """Multi-disease pipe-separated string should encode multiple positives."""
        from data.dataset import encode_labels
        result = encode_labels("Atelectasis|Effusion|Edema")
        assert result.sum() == 3.0, "Should have 3 positive labels"
        # Atelectasis=0, Effusion=2, Edema=9
        assert result[0] == 1.0  # Atelectasis
        assert result[2] == 1.0  # Effusion
        assert result[9] == 1.0  # Edema

    def test_label_encoding_unknown_disease_ignored(self):
        """Unknown disease labels should be silently ignored."""
        from data.dataset import encode_labels
        result = encode_labels("Atelectasis|UnknownDisease")
        assert result[0] == 1.0   # Atelectasis present
        assert result.sum() == 1.0  # Unknown ignored


# ─────────────────────────────────────────────────────────────────────────────
# Model Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestChestViT:
    """Tests that use the actual ViT model — requires ~330MB download on first run."""

    @pytest.fixture(scope="class")
    def model(self):
        """Load ViT model once for all tests in this class."""
        from models.vit_model import ChestViT
        # Use gradient_checkpointing=False for CPU testing
        m = ChestViT(num_classes=14, gradient_checkpointing=False)
        m.eval()
        return m

    def test_output_logits_shape(self, model):
        """Model should output (B, 14) logits."""
        batch = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits, attentions = model(batch, output_attentions=False)
        assert logits.shape == (2, 14), f"Expected (2,14), got {logits.shape}"

    def test_attention_weights_shape(self, model):
        """Attention weights should be 12 tensors of (B, 12, 197, 197)."""
        batch = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits, attentions = model(batch, output_attentions=True)
        assert attentions is not None, "Attentions should not be None"
        assert len(attentions) == 12, f"Expected 12 attention layers, got {len(attentions)}"
        for i, attn in enumerate(attentions):
            assert attn.shape == (1, 12, 197, 197), \
                f"Layer {i}: expected (1,12,197,197), got {attn.shape}"

    def test_sigmoid_output_range(self, model):
        """Sigmoid probabilities should be in (0, 1)."""
        batch = torch.randn(3, 3, 224, 224)
        with torch.no_grad():
            logits, _ = model(batch, output_attentions=False)
        probs = torch.sigmoid(logits)
        assert (probs > 0).all() and (probs < 1).all(), "Probs must be in (0,1)"

    def test_no_nans_in_output(self, model):
        """Model output should never contain NaN values."""
        batch = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits, _ = model(batch, output_attentions=False)
        assert not torch.isnan(logits).any(), "NaN detected in logits!"

    def test_parameter_count(self, model):
        """ViT-Base-16 should have ~86M parameters."""
        total = model.count_parameters()
        # ViT-Base has ~86M params; with 14-class head it's ~86M + tiny
        assert 80_000_000 < total < 90_000_000, \
            f"Unexpected parameter count: {total:,}"


# ─────────────────────────────────────────────────────────────────────────────
# Attention Rollout Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAttentionRollout:
    """Tests for the attention rollout explainability module."""

    def _make_fake_attentions(self, batch=1, heads=12, seq=197):
        """Generate synthetic attention tensors for testing."""
        attentions = []
        for _ in range(12):
            # Random attention weights, normalized (rows sum to 1)
            attn = torch.rand(batch, heads, seq, seq)
            attn = attn / attn.sum(dim=-1, keepdim=True)
            attentions.append(attn)
        return attentions

    def test_rollout_output_shape(self):
        """Rollout should return (B, 14, 14) patch attention map."""
        from explainability.attention_rollout import compute_attention_rollout
        attentions = self._make_fake_attentions(batch=2)
        rollout = compute_attention_rollout(attentions, discard_ratio=0.0)
        assert rollout.shape == (2, 14, 14), \
            f"Expected (2,14,14), got {rollout.shape}"

    def test_rollout_value_range(self):
        """Rollout values should be in [0, 1] after normalization."""
        from explainability.attention_rollout import compute_attention_rollout
        attentions = self._make_fake_attentions(batch=1)
        rollout = compute_attention_rollout(attentions, discard_ratio=0.0)
        assert rollout.min() >= 0.0 and rollout.max() <= 1.0, \
            f"Values out of [0,1] range: min={rollout.min()}, max={rollout.max()}"

    def test_rollout_head_fusions(self):
        """All three head fusion modes should return valid results."""
        from explainability.attention_rollout import compute_attention_rollout
        attentions = self._make_fake_attentions(batch=1)
        for fusion in ["mean", "max", "min"]:
            rollout = compute_attention_rollout(attentions, head_fusion=fusion, discard_ratio=0.0)
            assert rollout.shape == (1, 14, 14), f"Failed with head_fusion={fusion}"
            assert not np.isnan(rollout).any(), f"NaN with head_fusion={fusion}"

    def test_heatmap_overlay_shape(self):
        """Overlay should return (224, 224, 3) uint8 numpy array."""
        from explainability.attention_rollout import rollout_to_heatmap
        rollout = np.random.rand(14, 14).astype(np.float32)
        original = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        overlay = rollout_to_heatmap(rollout, original)
        assert overlay.shape == (224, 224, 3), f"Expected (224,224,3), got {overlay.shape}"
        assert overlay.dtype == np.uint8


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Metrics Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluationMetrics:
    def test_perfect_auc(self):
        """Perfect predictions should give AUC = 1.0."""
        from training.evaluate import compute_per_class_auc
        from data.dataset import DISEASE_LABELS
        # Perfect classifier: probs == labels
        labels = np.zeros((100, 14), dtype=np.float32)
        labels[:50, 0] = 1.0   # Atelectasis positive in first 50
        probs = labels.copy()  # Perfect predictions
        aucs = compute_per_class_auc(probs, labels, DISEASE_LABELS)
        assert abs(aucs["Atelectasis"] - 1.0) < 1e-6, \
            f"Expected AUC=1.0 for Atelectasis, got {aucs['Atelectasis']}"

    def test_random_auc_near_half(self):
        """Random predictions should give AUC near 0.5."""
        from training.evaluate import compute_per_class_auc
        from data.dataset import DISEASE_LABELS
        np.random.seed(42)
        labels = (np.random.rand(2000, 14) > 0.8).astype(np.float32)
        probs  = np.random.rand(2000, 14).astype(np.float32)
        aucs   = compute_per_class_auc(probs, labels, DISEASE_LABELS)
        valid_aucs = [v for v in aucs.values() if not np.isnan(v)]
        macro = np.mean(valid_aucs)
        assert 0.4 < macro < 0.6, \
            f"Random classifier AUC should be ~0.5, got {macro:.3f}"

    def test_compute_metrics_keys(self):
        """compute_metrics should return all expected keys."""
        from training.evaluate import compute_metrics
        from data.dataset import DISEASE_LABELS
        probs  = np.random.rand(100, 14).astype(np.float32)
        labels = (np.random.rand(100, 14) > 0.8).astype(np.float32)
        metrics = compute_metrics(probs, labels, DISEASE_LABELS)
        expected_keys = [
            "per_class_auc", "macro_auc",
            "per_class_ap",  "macro_ap",
            "per_class_threshold_metrics",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# Loss Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLosses:
    def test_weighted_bce_output_shape(self):
        """Loss should return a scalar."""
        from training.losses import WeightedBCEWithLogitsLoss
        weights = torch.ones(14)
        loss_fn = WeightedBCEWithLogitsLoss(pos_weight=weights)
        logits  = torch.randn(8, 14)
        targets = (torch.rand(8, 14) > 0.8).float()
        loss = loss_fn(logits, targets)
        assert loss.shape == torch.Size([]), f"Loss should be scalar, got {loss.shape}"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert loss.item() > 0, "Loss should be positive"

    def test_focal_loss_scalar(self):
        """Focal loss should return a scalar."""
        from training.losses import FocalLoss
        loss_fn = FocalLoss(gamma=2.0)
        logits  = torch.randn(8, 14)
        targets = (torch.rand(8, 14) > 0.8).float()
        loss = loss_fn(logits, targets)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
