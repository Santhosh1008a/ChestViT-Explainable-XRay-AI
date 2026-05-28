"""
training/evaluate.py
---------------------
Evaluation utilities: AUC-ROC per class + aggregate metrics.

Medical AI convention:
  - Report AUC-ROC per disease class (not accuracy — useless with imbalanced data)
  - Report macro-averaged AUC (average of 14 per-class AUCs)
  - Compare against the NIH ChestX-ray14 paper baseline (~0.74 macro AUC)
    and CheXNet (DenseNet-121, ~0.84 macro AUC)

AUC-ROC interpretation:
  0.5  = random classifier
  0.70 = reasonable
  0.80 = good (approaching radiologist level for some conditions)
  0.90 = excellent
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm


# NIH baseline AUC-ROC from Wang et al. 2017 (for reference)
NIH_BASELINE_AUC = {
    "Atelectasis":       0.7003,
    "Cardiomegaly":      0.8100,
    "Effusion":          0.7585,
    "Infiltration":      0.6614,
    "Mass":              0.6933,
    "Nodule":            0.6689,
    "Pneumonia":         0.6580,
    "Pneumothorax":      0.7993,
    "Consolidation":     0.7032,
    "Edema":             0.8052,
    "Emphysema":         0.8330,
    "Fibrosis":          0.7859,
    "Pleural_Thickening":0.6835,
    "Hernia":            0.8717,
}


@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    desc: str = "Evaluating",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a dataloader and collect all predictions and labels.

    Args:
        model:      ChestViT in eval mode.
        dataloader: Validation or test DataLoader.
        device:     Target device.
        desc:       tqdm progress bar description.

    Returns:
        all_probs:  (N, 14) float32 numpy array of sigmoid probabilities.
        all_labels: (N, 14) float32 numpy array of ground-truth multi-hot labels.
    """
    model.eval()
    all_probs  = []
    all_labels = []

    for batch in tqdm(dataloader, desc=desc, leave=False):
        images, labels, _ = batch
        images = images.to(device, non_blocking=True)

        # No need for attention during evaluation — faster
        logits, _ = model(images, output_attentions=False)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return np.vstack(all_probs), np.vstack(all_labels)


def compute_per_class_auc(
    probs: np.ndarray,
    labels: np.ndarray,
    disease_names: List[str],
) -> Dict[str, float]:
    """
    Compute AUC-ROC for each of the 14 disease classes.

    Handles edge cases:
      - Classes with zero positive samples get AUC = NaN (excluded from macro average)

    Args:
        probs:         (N, 14) predicted probabilities.
        labels:        (N, 14) ground-truth binary labels.
        disease_names: List of 14 disease label strings.

    Returns:
        Dict mapping disease_name → AUC-ROC score.
    """
    aucs = {}
    for i, disease in enumerate(disease_names):
        n_pos = labels[:, i].sum()
        n_neg = (1 - labels[:, i]).sum()
        if n_pos == 0 or n_neg == 0:
            aucs[disease] = float("nan")
            continue
        try:
            aucs[disease] = roc_auc_score(labels[:, i], probs[:, i])
        except Exception:
            aucs[disease] = float("nan")
    return aucs


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    disease_names: List[str],
    threshold: float = 0.5,
) -> Dict:
    """
    Compute the full set of evaluation metrics.

    Returns:
        Dict with keys:
          - "per_class_auc": dict of disease → AUC
          - "macro_auc":     mean AUC across classes (ignoring NaN)
          - "per_class_ap":  dict of disease → Average Precision (AUPRC)
          - "macro_ap":      mean AP across classes
          - "per_class_threshold_metrics": dict of disease → {sensitivity, specificity, f1}
    """
    per_class_auc = compute_per_class_auc(probs, labels, disease_names)
    valid_aucs = [v for v in per_class_auc.values() if not np.isnan(v)]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

    per_class_ap = {}
    for i, disease in enumerate(disease_names):
        try:
            per_class_ap[disease] = average_precision_score(labels[:, i], probs[:, i])
        except Exception:
            per_class_ap[disease] = float("nan")

    valid_aps = [v for v in per_class_ap.values() if not np.isnan(v)]
    macro_ap = float(np.mean(valid_aps)) if valid_aps else float("nan")

    # Threshold-based metrics (sensitivity, specificity, F1)
    preds = (probs >= threshold).astype(float)
    per_class_threshold_metrics = {}
    for i, disease in enumerate(disease_names):
        tp = ((preds[:, i] == 1) & (labels[:, i] == 1)).sum()
        fp = ((preds[:, i] == 1) & (labels[:, i] == 0)).sum()
        fn = ((preds[:, i] == 0) & (labels[:, i] == 1)).sum()
        tn = ((preds[:, i] == 0) & (labels[:, i] == 0)).sum()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        precision   = tp / (tp + fp + 1e-8)
        f1          = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)
        per_class_threshold_metrics[disease] = {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "f1":          float(f1),
        }

    return {
        "per_class_auc":               per_class_auc,
        "macro_auc":                   macro_auc,
        "per_class_ap":                per_class_ap,
        "macro_ap":                    macro_ap,
        "per_class_threshold_metrics": per_class_threshold_metrics,
    }


def print_results_table(
    metrics: Dict,
    disease_names: List[str],
    compare_baseline: bool = True,
) -> None:
    """Print a formatted results table to console."""
    print("\n" + "═" * 72)
    print(f"{'Disease':<22} {'AUC-ROC':>8} {'NIH Base':>10} {'Δ AUC':>8} {'AP':>8} {'F1':>8}")
    print("─" * 72)

    auc_d   = metrics["per_class_auc"]
    ap_d    = metrics["per_class_ap"]
    thresh_d = metrics["per_class_threshold_metrics"]

    for disease in disease_names:
        auc     = auc_d.get(disease, float("nan"))
        ap      = ap_d.get(disease, float("nan"))
        f1      = thresh_d.get(disease, {}).get("f1", float("nan"))
        baseline = NIH_BASELINE_AUC.get(disease, float("nan"))
        delta   = auc - baseline if not (np.isnan(auc) or np.isnan(baseline)) else float("nan")

        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "   N/A"
        print(
            f"{disease:<22} "
            f"{auc:>8.4f} "
            f"{baseline:>10.4f} "
            f"{delta_str:>8} "
            f"{ap:>8.4f} "
            f"{f1:>8.4f}"
        )

    print("─" * 72)
    macro_auc = metrics["macro_auc"]
    macro_ap  = metrics["macro_ap"]
    nih_macro = float(np.mean(list(NIH_BASELINE_AUC.values())))
    delta_macro = macro_auc - nih_macro
    print(
        f"{'MACRO AVERAGE':<22} "
        f"{macro_auc:>8.4f} "
        f"{nih_macro:>10.4f} "
        f"{delta_macro:>+8.4f} "
        f"{macro_ap:>8.4f}"
    )
    print("═" * 72)


def plot_roc_curves(
    probs: np.ndarray,
    labels: np.ndarray,
    disease_names: List[str],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot per-class ROC curves in a grid layout.

    Args:
        probs:         (N, 14) probabilities.
        labels:        (N, 14) binary labels.
        disease_names: List of 14 disease names.
        save_path:     Optional path to save figure.

    Returns:
        matplotlib Figure.
    """
    n = len(disease_names)
    cols = 4
    rows = (n + cols - 1) // cols  # = 4

    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    fig.patch.set_facecolor("#0f172a")
    fig.suptitle("ROC Curves — Per Disease Class", color="white",
                 fontsize=15, fontweight="bold")

    axes_flat = axes.flatten()

    for i, (ax, disease) in enumerate(zip(axes_flat, disease_names)):
        ax.set_facecolor("#1e293b")
        n_pos = labels[:, i].sum()
        if n_pos == 0:
            ax.text(0.5, 0.5, "No positives\nin test set",
                    ha="center", va="center", color="gray", fontsize=9)
            ax.set_title(disease, color="gray", fontsize=9)
            continue

        fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
        auc = roc_auc_score(labels[:, i], probs[:, i])
        baseline = NIH_BASELINE_AUC.get(disease, None)

        ax.plot(fpr, tpr, color="#60a5fa", lw=2,
                label=f"ViT AUC={auc:.3f}")
        if baseline:
            ax.axhline(y=baseline, color="#f97316", linestyle="--", lw=1, alpha=0.7,
                       label=f"NIH={baseline:.3f}")
        ax.plot([0, 1], [0, 1], color="#475569", linestyle=":", lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_title(disease, color="white", fontsize=9, fontweight="bold")
        ax.tick_params(colors="#94a3b8", labelsize=7)
        ax.legend(fontsize=7, loc="lower right",
                  facecolor="#0f172a", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    # Hide extra subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  ROC curves saved → {save_path}")

    return fig


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    disease_names: List[str],
    results_dir: Optional[Path] = None,
    split: str = "test",
) -> Dict:
    """
    Full evaluation pipeline: inference → metrics → print → plots.

    Args:
        model:         Trained ChestViT model.
        dataloader:    Test or val DataLoader.
        device:        Target device.
        disease_names: List of 14 disease names.
        results_dir:   Directory to save result plots/CSV.
        split:         "val" or "test" — used for filenames.

    Returns:
        Full metrics dictionary.
    """
    print(f"\n{'='*50}")
    print(f"  Evaluating on {split.upper()} set...")

    probs, labels = run_inference(model, dataloader, device, desc=f"Eval ({split})")
    metrics = compute_metrics(probs, labels, disease_names)
    print_results_table(metrics, disease_names)

    if results_dir:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save ROC curves
        plot_roc_curves(
            probs, labels, disease_names,
            save_path=results_dir / f"roc_curves_{split}.png",
        )

        # Save metrics as CSV
        import pandas as pd
        rows = []
        for disease in disease_names:
            rows.append({
                "disease":     disease,
                "auc_roc":     metrics["per_class_auc"].get(disease, float("nan")),
                "avg_prec":    metrics["per_class_ap"].get(disease, float("nan")),
                "f1":          metrics["per_class_threshold_metrics"].get(disease, {}).get("f1", float("nan")),
                "sensitivity": metrics["per_class_threshold_metrics"].get(disease, {}).get("sensitivity", float("nan")),
                "specificity": metrics["per_class_threshold_metrics"].get(disease, {}).get("specificity", float("nan")),
                "nih_baseline": NIH_BASELINE_AUC.get(disease, float("nan")),
            })
        df = pd.DataFrame(rows)
        df.to_csv(results_dir / f"metrics_{split}.csv", index=False)
        print(f"  Metrics CSV saved → {results_dir / f'metrics_{split}.csv'}")

    return metrics
