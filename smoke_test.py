"""
smoke_test.py
-------------
Quick sanity check — verifies the entire pipeline with SYNTHETIC data.
Does NOT require the NIH dataset or a GPU.

Run this BEFORE downloading the dataset to confirm your environment is working:
    python smoke_test.py

Checks:
  ✓ All imports succeed
  ✓ CLAHE preprocessing works
  ✓ ViT model loads and produces correct output shapes
  ✓ Attention rollout runs end-to-end
  ✓ Loss functions compute without NaN
  ✓ Gradio app imports without errors
"""

import sys
import os
import traceback
from pathlib import Path

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

def section(title: str):
    print(f"\n" + "-"*60)
    print(f"  {title}")
    print("-"*60)

def ok(msg: str):
    print(f"  [OK] {msg}")

def fail(msg: str, exc=None):
    print(f"  [FAIL] {msg}")
    if exc:
        traceback.print_exc()

def main():
    print("\n" + "="*60)
    print("  ChestViT Smoke Test -- Environment Verification")
    print("="*60)

    errors = 0

    # ── 1. Core imports ───────────────────────────────────────────────────────
    section("1. Core Imports")
    try:
        import torch
        import numpy as np
        import cv2
        import albumentations
        import transformers
        import sklearn
        import mlflow
        import gradio
        ok(f"PyTorch {torch.__version__} — CUDA: {torch.cuda.is_available()}")
        ok(f"Transformers {transformers.__version__}")
        ok(f"OpenCV {cv2.__version__}")
        ok(f"Gradio {gradio.__version__}")
        ok(f"MLflow {mlflow.__version__}")
    except ImportError as e:
        fail(f"Import failed: {e}")
        errors += 1

    # ── 2. Preprocessing ──────────────────────────────────────────────────────
    section("2. Preprocessing Pipeline")
    try:
        import numpy as np
        from data.preprocessing import apply_clahe, get_train_transforms, get_val_transforms

        gray = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        clahe_result = apply_clahe(gray)
        ok(f"CLAHE: {gray.shape} → {clahe_result.shape}")

        dummy_rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        train_t = get_train_transforms(224)
        tensor = train_t(image=dummy_rgb)["image"]
        ok(f"Train transform output: {tensor.shape} dtype={tensor.dtype}")
    except Exception as e:
        fail("Preprocessing failed", e)
        errors += 1

    # ── 3. Label Encoding ─────────────────────────────────────────────────────
    section("3. Dataset Label Encoding")
    try:
        from data.dataset import encode_labels, DISEASE_LABELS
        v1 = encode_labels("No Finding")
        v2 = encode_labels("Atelectasis|Effusion")
        assert v1.sum() == 0, "No Finding should be all zeros"
        assert v2.sum() == 2, "Two diseases should give 2 positives"
        ok(f"Label encoding correct ({len(DISEASE_LABELS)} diseases)")
    except Exception as e:
        fail("Label encoding failed", e)
        errors += 1

    # ── 4. ViT Model ──────────────────────────────────────────────────────────
    section("4. ViT-Base-16 Model (downloads weights if first run)")
    try:
        import torch
        from models.vit_model import ChestViT
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = ChestViT(num_classes=14, gradient_checkpointing=False)
        model.to(device).eval()

        dummy = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            logits, attentions = model(dummy, output_attentions=True)

        assert logits.shape == (1, 14), f"Logits shape: {logits.shape}"
        assert len(attentions) == 12, f"Attention layers: {len(attentions)}"
        assert attentions[0].shape == (1, 12, 197, 197)
        ok(f"Logits shape: {logits.shape}")
        ok(f"Attention: {len(attentions)} layers × {attentions[0].shape}")
        ok(f"Parameters: {model.count_parameters():,}")
    except Exception as e:
        fail("Model failed", e)
        errors += 1

    # ── 5. Attention Rollout ───────────────────────────────────────────────────
    section("5. Attention Rollout")
    try:
        import torch, numpy as np
        from explainability.attention_rollout import compute_attention_rollout, rollout_to_heatmap

        fake_attn = [
            torch.rand(1, 12, 197, 197) for _ in range(12)
        ]
        rollout = compute_attention_rollout(fake_attn, discard_ratio=0.9)
        assert rollout.shape == (1, 14, 14), f"Rollout shape: {rollout.shape}"
        ok(f"Rollout shape: {rollout.shape}")

        original = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        overlay = rollout_to_heatmap(rollout[0], original)
        assert overlay.shape == (224, 224, 3)
        ok(f"Heatmap overlay: {overlay.shape}")
    except Exception as e:
        fail("Attention rollout failed", e)
        errors += 1

    # ── 6. Loss Functions ─────────────────────────────────────────────────────
    section("6. Loss Functions")
    try:
        import torch
        from training.losses import WeightedBCEWithLogitsLoss, FocalLoss

        weights = torch.ones(14)
        logits  = torch.randn(4, 14)
        targets = (torch.rand(4, 14) > 0.8).float()

        bce_loss   = WeightedBCEWithLogitsLoss(weights)(logits, targets)
        focal_loss = FocalLoss(gamma=2.0)(logits, targets)

        assert not torch.isnan(bce_loss), "BCE loss is NaN!"
        assert not torch.isnan(focal_loss), "Focal loss is NaN!"
        ok(f"Weighted BCE Loss: {bce_loss.item():.4f}")
        ok(f"Focal Loss: {focal_loss.item():.4f}")
    except Exception as e:
        fail("Loss functions failed", e)
        errors += 1

    # ── 7. Gradio App Import ───────────────────────────────────────────────────
    section("7. Gradio App Import (DEMO_MODE=1)")
    try:
        import os
        os.environ["DEMO_MODE"] = "1"
        import importlib
        import app.gradio_app  # Should not crash with DEMO_MODE=1
        ok("Gradio app imports successfully")
    except Exception as e:
        fail("Gradio app import failed", e)
        errors += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    if errors == 0:
        print("  All checks passed! Environment is ready.")
        print("\n  Next steps:")
        print("  1. python data/download.py          # Download NIH dataset (~42 GB)")
        print("  2. python training/train.py          # Train the model")
        print("  3. python app/gradio_app.py          # Launch demo")
    else:
        print(f"  {errors} check(s) failed. See errors above.")
        print("  Run: pip install -r requirements.txt")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
