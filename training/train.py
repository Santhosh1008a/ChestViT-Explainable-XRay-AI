"""
training/train.py
-----------------
Main training loop for ChestViT on NIH ChestX-ray14.

RTX 3050 Optimizations (4 GB VRAM):
  ✓ Batch size 8 (fits ViT-Base-16 with grad-checkpointing)
  ✓ Gradient accumulation (effective batch = 32, no extra VRAM)
  ✓ Mixed precision (fp16 via torch.cuda.amp — ~2× speedup, ~50% VRAM)
  ✓ Gradient checkpointing (in model — ~30% VRAM reduction)
  ✓ Cosine LR schedule with warmup (better convergence than step decay)
  ✓ MLflow local tracking (no internet needed)

Estimated training time on RTX 3050:
  - 5 epochs on full 86K train images ≈ 8–12 hours
  - Set train_fraction=0.2 in config for a ~2-hour smoke test
"""

import os
import sys
import time
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import load_config
from data.dataset import build_dataloaders, DISEASE_LABELS
from models.vit_model import ChestViT, save_checkpoint, load_full_checkpoint
from training.losses import build_loss
from training.evaluate import run_inference, compute_metrics, print_results_table


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup → cosine annealing LR schedule.
    Standard schedule for ViT fine-tuning.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: ChestViT,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    loss_fn: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    log_interval: int,
    mlflow_run,
) -> Dict[str, float]:
    """
    Run a single training epoch.

    Returns:
        Dict with "train_loss" and "train_lr".
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(enumerate(dataloader),
                desc=f"Epoch {epoch:02d} [train]", leave=True)

    global_step = (epoch - 1) * 1000 # Dummy value since we cannot easily get len(dataloader) for stream

    for step, (images, labels, _) in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── Forward pass (mixed precision) ────────────────────────────────────
        with autocast(enabled=True):
            logits, _ = model(images, output_attentions=False)
            loss = loss_fn(logits, labels)
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

        # ── Backward pass (scaled) ────────────────────────────────────────────
        scaler.scale(loss).backward()

        # ── Gradient accumulation step ────────────────────────────────────────
        if (step + 1) % gradient_accumulation_steps == 0:
            # Unscale for gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # Track loss (scale back up for reporting)
        batch_loss = loss.item() * gradient_accumulation_steps
        total_loss += batch_loss
        num_batches += 1

        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            "loss": f"{batch_loss:.4f}",
            "lr": f"{current_lr:.2e}",
        })

        # ── MLflow logging ────────────────────────────────────────────────────
        if (step + 1) % log_interval == 0:
            mlflow.log_metrics({
                "train/step_loss": batch_loss,
                "train/lr": current_lr,
            }, step=global_step + step)

    avg_loss = total_loss / max(1, num_batches)
    return {"train_loss": avg_loss, "train_lr": scheduler.get_last_lr()[0]}


def validate(
    model: ChestViT,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    disease_names,
) -> Dict[str, float]:
    """
    Run validation and compute loss + AUC metrics.

    Returns:
        Dict with "val_loss", "val_macro_auc", and per-class AUCs.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for images, labels, _ in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=True):
                logits, _ = model(images, output_attentions=False)
                loss = loss_fn(logits, labels)

            total_loss += loss.item()
            num_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    probs_arr  = np.vstack(all_probs)
    labels_arr = np.vstack(all_labels)
    avg_loss   = total_loss / max(1, num_batches)

    metrics = compute_metrics(probs_arr, labels_arr, disease_names)

    result = {
        "val_loss":      avg_loss,
        "val_macro_auc": metrics["macro_auc"],
    }
    result.update({f"val_auc_{d}": metrics["per_class_auc"][d]
                   for d in disease_names})
    return result


def train(config_path: Optional[str] = None, resume_from: Optional[str] = None) -> None:
    """
    Main training entry point.
    Reads config from config/config.yaml (or custom path).
    """
    from config_loader import CFG as global_cfg
    if config_path:
        cfg = load_config(config_path)
    else:
        cfg = global_cfg


    # ── Device ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n  ⚠ No GPU detected — training on CPU (very slow!)")
        print("  Tip: Ensure CUDA drivers and PyTorch CUDA are installed.")

    # ── Data ───────────────────────────────────────────────────────────────────
    print("\n[1/5] Building DataLoaders...")
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        image_size      = cfg.dataset.image_size,
        batch_size      = cfg.training.batch_size,
        num_workers     = cfg.dataset.num_workers,
        pin_memory      = cfg.dataset.pin_memory and device.type == "cuda",
        val_split       = cfg.dataset.val_split,
        train_fraction  = cfg.dataset.train_fraction,
        train_take_limit= getattr(cfg.dataset, 'train_take_limit', 20000),
        val_take_limit  = getattr(cfg.dataset, 'val_take_limit', 2000),
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    print("\n[2/5] Building Model...")
    model = ChestViT(
        num_classes           = cfg.model.num_classes,
        pretrained_name       = cfg.model.name,
        dropout               = cfg.model.dropout,
        gradient_checkpointing= cfg.model.gradient_checkpointing,
    ).to(device)

    # ── Loss ───────────────────────────────────────────────────────────────────
    print("\n[3/5] Building Loss...")
    loss_fn = build_loss("weighted_bce", class_weights, device)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    print("\n[4/5] Building Optimizer + Scheduler...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.training.learning_rate,
        weight_decay = cfg.training.weight_decay,
    )

    total_samples_per_epoch = cfg.dataset.train_take_limit if getattr(cfg.dataset, 'train_take_limit', None) else 89696 # BahaaEldin0 train size
    batches_per_epoch = total_samples_per_epoch // cfg.training.batch_size
    total_steps  = batches_per_epoch * cfg.training.num_epochs // cfg.training.gradient_accumulation_steps
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler(enabled=cfg.training.mixed_precision)

    print(f"  Total optimizer steps : {total_steps:,}")
    print(f"  Warmup steps          : {warmup_steps:,}")
    print(f"  Mixed precision (fp16): {cfg.training.mixed_precision}")

    # ── MLflow ─────────────────────────────────────────────────────────────────
    print("\n[5/5] Starting MLflow tracking...")
    Path(cfg.paths.mlflow_dir).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{Path(cfg.paths.mlflow_dir).resolve()}")
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    checkpoints_dir = Path(cfg.paths.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Training Loop ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Starting training — {cfg.training.num_epochs} epochs")
    print(f"  Effective batch size: {cfg.training.batch_size * cfg.training.gradient_accumulation_steps}")
    print(f"{'='*60}\n")

    best_val_auc   = -float("inf")
    best_ckpt_path = checkpoints_dir / "best_model.pt"
    start_epoch    = 1

    if resume_from:
        start_epoch, best_val_auc = load_full_checkpoint(
            checkpoint_path=resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
        )

    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        # Log all hyperparameters
        mlflow.log_params({
            "model":                    cfg.model.name,
            "num_classes":              cfg.model.num_classes,
            "batch_size":               cfg.training.batch_size,
            "gradient_accumulation":    cfg.training.gradient_accumulation_steps,
            "effective_batch_size":     cfg.training.batch_size * cfg.training.gradient_accumulation_steps,
            "learning_rate":            cfg.training.learning_rate,
            "weight_decay":             cfg.training.weight_decay,
            "num_epochs":               cfg.training.num_epochs,
            "mixed_precision":          cfg.training.mixed_precision,
            "gradient_checkpointing":   cfg.model.gradient_checkpointing,
            "train_fraction":           cfg.dataset.train_fraction,
            "warmup_ratio":             cfg.training.warmup_ratio,
        })

        for epoch in range(start_epoch, cfg.training.num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                scaler=scaler,
                device=device,
                epoch=epoch,
                gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
                max_grad_norm=cfg.training.max_grad_norm,
                log_interval=cfg.training.log_interval,
                mlflow_run=run,
            )

            # Validate
            val_metrics = validate(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
                disease_names=DISEASE_LABELS,
            )
            # After full validation on last epoch, generate artifacts
            if epoch == cfg.training.num_epochs:
                from training.evaluate import evaluate_model

                # Small hack to reload the subset dataloader since it's consumed
                # (Huggingface streams cannot be reset easily in all scenarios without recreating)
                _, val_loader2, _, _ = build_dataloaders(
                    image_size      = cfg.dataset.image_size,
                    batch_size      = cfg.training.batch_size,
                    num_workers     = cfg.dataset.num_workers,
                    pin_memory      = cfg.dataset.pin_memory and device.type == "cuda",
                    val_split       = cfg.dataset.val_split,
                    train_fraction  = cfg.dataset.train_fraction,
                    train_take_limit= getattr(cfg.dataset, 'train_take_limit', 20000),
                    val_take_limit  = getattr(cfg.dataset, 'val_take_limit', 2000),
                )

                evaluate_model(
                    model=model,
                    dataloader=val_loader2,
                    device=device,
                    disease_names=DISEASE_LABELS,
                    results_dir=results_dir,
                    split="val"
                )

            epoch_time = time.time() - epoch_start

            # Print epoch summary
            print(f"\n  Epoch {epoch:02d}/{cfg.training.num_epochs} "
                  f"| Time: {epoch_time/60:.1f}m "
                  f"| Train Loss: {train_metrics['train_loss']:.4f} "
                  f"| Val Loss: {val_metrics['val_loss']:.4f} "
                  f"| Val AUC: {val_metrics['val_macro_auc']:.4f}")

            # Log to MLflow
            mlflow.log_metrics({
                "epoch/train_loss":  train_metrics["train_loss"],
                "epoch/val_loss":    val_metrics["val_loss"],
                "epoch/val_macro_auc": val_metrics["val_macro_auc"],
                **{f"epoch/val_auc_{d}": val_metrics[f"val_auc_{d}"]
                   for d in DISEASE_LABELS},
            }, step=epoch)

            # Save best checkpoint
            if val_metrics["val_macro_auc"] > best_val_auc:
                best_val_auc = val_metrics["val_macro_auc"]
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_auc=best_val_auc,
                    save_path=best_ckpt_path,
                )
                mlflow.log_metric("best_val_auc", best_val_auc, step=epoch)

            # Always save latest
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_auc=val_metrics["val_macro_auc"],
                save_path=checkpoints_dir / f"epoch_{epoch:02d}.pt",
            )

        print(f"\n{'='*60}")
        print(f"  Training complete! Best Val AUC: {best_val_auc:.4f}")
        print(f"  Best checkpoint: {best_ckpt_path}")
        print(f"  MLflow UI: mlflow ui --backend-store-uri {Path(cfg.paths.mlflow_dir).resolve()}")
        print(f"{'='*60}\n")

        # Log best model artifact
        mlflow.log_artifact(str(best_ckpt_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ChestViT on NIH ChestX-ray14")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml (default: config/config.yaml)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()
    train(args.config, resume_from=args.resume)
