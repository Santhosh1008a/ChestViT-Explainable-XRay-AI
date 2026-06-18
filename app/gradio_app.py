"""
app/gradio_app.py
------------------
Gradio demo dashboard for ChestViT — Explainable Chest X-Ray Analysis.

Features:
  ┌────────────────────────────────────────────────────────────────┐
  │  Upload X-Ray  │  Attention Rollout Heatmap Overlay           │
  ├────────────────────────────────────────────────────────────────┤
  │  CLAHE Preview │  Disease Probability Bar Chart (14 classes)  │
  └────────────────────────────────────────────────────────────────┘

  + Top diagnoses summary text
  + Model info sidebar
  + MLflow metrics link

Run locally:
  python app/gradio_app.py

Requirements:
  - Trained model checkpoint at checkpoints/best_model.pt
  - OR set DEMO_MODE=1 to run with random weights for UI preview
"""

import os
import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_download
import torch

# Fix Windows console encoding (cp1252 can't handle emoji/Unicode)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import gradio as gr
from explainability.gradcam import generate_gradcam_heatmap
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import load_config
from data.preprocessing import load_and_preprocess_raw, get_val_transforms, denormalize, apply_clahe
from models.vit_model import ChestViT, load_checkpoint
from explainability.attention_rollout import explain_prediction, rollout_to_heatmap
from data.dataset import DISEASE_LABELS

# ── Config ────────────────────────────────────────────────────────────────────
cfg = load_config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ── CSS Styling ───────────────────────────────────────────────────────────────
CUSTOM_CSS = """
:root {
    --primary:    #6366f1;
    --primary-dark: #4f46e5;
    --surface:    #1e1e2e;
    --surface-2:  #2a2a3e;
    --text:       #e2e8f0;
    --text-muted: #94a3b8;
    --accent:     #22d3ee;
    --danger:     #ef4444;
    --warning:    #f97316;
    --success:    #22c55e;
    --border:     rgba(99, 102, 241, 0.25);
}

body, .gradio-container {
    background: #0f0f1a !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

.gr-form, .gr-panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.2s ease !important;
}
.gr-button-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
}

label, .label-wrap span {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

h1, h2, h3 { color: var(--text) !important; }

.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #22d3ee, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem;
}
.header-sub {
    color: var(--text-muted);
    text-align: center;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}
.stat-card {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 16px;
    margin: 4px;
    text-align: center;
}
"""

HEADER_HTML = """
<div style="text-align:center; padding: 20px 0 10px 0;">
    <div style="font-size:2.4rem; font-weight:800; background:linear-gradient(135deg,#6366f1,#22d3ee,#22c55e);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        🫁 ChestViT — Explainable X-Ray AI
    </div>
    <div style="color:#94a3b8; font-size:0.95rem; margin-top:8px;">
        ViT-Base-16 · 14-Disease Multi-Label Classification · Attention Rollout Explainability
    </div>
    <div style="display:flex; justify-content:center; gap:16px; margin-top:14px; flex-wrap:wrap;">
        <span style="background:#1e1e2e; border:1px solid rgba(99,102,241,0.3); border-radius:8px;
                     padding:6px 14px; color:#a5b4fc; font-size:0.82rem; font-weight:600;">
            🤖 google/vit-base-patch16-224-in21k
        </span>
        <span style="background:#1e1e2e; border:1px solid rgba(34,211,238,0.3); border-radius:8px;
                     padding:6px 14px; color:#67e8f9; font-size:0.82rem; font-weight:600;">
            📊 NIH ChestX-ray14 Dataset
        </span>
        <span style="background:#1e1e2e; border:1px solid rgba(34,197,94,0.3); border-radius:8px;
                     padding:6px 14px; color:#86efac; font-size:0.82rem; font-weight:600;">
            🔥 Attention Rollout XAI
        </span>
    </div>
</div>
"""

FOOTER_HTML = """
<div style="text-align:center; color:#475569; font-size:0.8rem; padding:16px 0 8px 0; border-top:1px solid rgba(99,102,241,0.15); margin-top:16px;">
    ⚠️ <strong>Research / Educational Use Only.</strong>
    This tool is NOT a medical device and should NOT be used for clinical diagnosis.
    Always consult a qualified radiologist.
</div>
"""


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model() -> ChestViT:
    """Load trained model from Hugging Face Model Hub."""

    ckpt_path = hf_hub_download(
        repo_id="sandy45/ChestViT-ViTBase-NIH14",
        filename="best_model.pt"
    )

    model = load_checkpoint(ckpt_path, DEVICE)

    model.to(DEVICE)
    model.eval()

    print("🔥 Loaded trained model from Hugging Face Hub")

    return model


# Load model once at startup
print(f"\nLoading model on {DEVICE}...")
MODEL = load_model()
VAL_TRANSFORM = get_val_transforms(cfg.dataset.image_size)
print("Model ready.\n")


# ── Inference Pipeline ────────────────────────────────────────────────────────

def preprocess_uploaded_image(pil_image: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
    """
    Convert a PIL image (from Gradio upload) to:
      1. CLAHE-enhanced numpy array for display
      2. Normalized tensor for model input

    Returns:
        (clahe_rgb, input_tensor) where:
          clahe_rgb: (H, W, 3) uint8 numpy array
          input_tensor: (1, 3, 224, 224) float32 tensor
    """
    # Convert to numpy
    img_np = np.array(pil_image.convert("RGB"))

    # To grayscale → CLAHE → back to RGB
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    if gray.dtype == np.uint16:
        gray = (gray / 256).astype(np.uint8)
    clahe_gray = apply_clahe(gray, clip_limit=2.0, tile_size=8)
    clahe_gray = cv2.resize(clahe_gray, (224, 224), interpolation=cv2.INTER_AREA)
    clahe_rgb = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2RGB)

    # Normalize for model
    augmented = VAL_TRANSFORM(image=clahe_rgb)
    tensor = augmented["image"].unsqueeze(0)  # (1, 3, 224, 224)

    return clahe_rgb, tensor


def make_probability_figure(probs: np.ndarray, threshold: float = 0.5) -> plt.Figure:
    """
    Create a beautiful dark-themed horizontal bar chart of disease probabilities.
    """
    sorted_idx = np.argsort(probs)  # ascending for bottom-to-top barh
    sorted_probs = probs[sorted_idx]
    sorted_names = [DISEASE_LABELS[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    # Color code by probability
    colors = []
    for p in sorted_probs:
        if p >= 0.7:
            colors.append("#ef4444")   # Red — high confidence positive
        elif p >= 0.5:
            colors.append("#f97316")   # Orange — positive
        elif p >= 0.3:
            colors.append("#eab308")   # Yellow — uncertain
        else:
            colors.append("#3b82f6")   # Blue — likely negative

    bars = ax.barh(range(len(sorted_names)), sorted_probs, color=colors,
                   edgecolor="none", height=0.65)

    # Threshold line
    ax.axvline(x=threshold, color="#a855f7", linestyle="--", linewidth=1.5,
               alpha=0.8, label=f"Threshold ({threshold})")

    # Labels
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, color="#e2e8f0", fontsize=9.5)
    ax.set_xlabel("Probability", color="#94a3b8", fontsize=10)
    ax.set_title("Disease Probability Scores", color="white",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0, 1.0)
    ax.tick_params(axis="x", colors="#94a3b8", labelsize=9)

    # Value labels on bars
    for bar, p in zip(bars, sorted_probs):
        ax.text(min(p + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{p:.3f}", va="center", color="white", fontsize=8.5, fontweight="bold")

    # Legend
    patches = [
        mpatches.Patch(color="#ef4444", label="High confidence (≥0.7)"),
        mpatches.Patch(color="#f97316", label="Positive (≥0.5)"),
        mpatches.Patch(color="#eab308", label="Uncertain (0.3–0.5)"),
        mpatches.Patch(color="#3b82f6", label="Likely negative (<0.3)"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=7.5,
              facecolor="#0f0f1a", labelcolor="white", framealpha=0.8)

    for spine in ax.spines.values():
        spine.set_edgecolor("#2d2d4e")

    plt.tight_layout()
    return fig


def make_heatmap_figure(
    clahe_rgb: np.ndarray,
    rollout: np.ndarray,
    overlay: np.ndarray,
) -> plt.Figure:
    """
    3-panel figure: original | raw rollout | overlay.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.patch.set_facecolor("#0f0f1a")

    titles = ["CLAHE-Enhanced X-Ray", "Attention Rollout Map", "Heatmap Overlay"]
    for ax, title in zip(axes, titles):
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=8)
        ax.axis("off")

    axes[0].imshow(clahe_rgb)
    axes[0].text(5, 218, "Input", color="#94a3b8", fontsize=8,
                 va="bottom", ha="left", fontweight="bold")

    rollout_display = cv2.resize(rollout, (224, 224), interpolation=cv2.INTER_CUBIC)
    im = axes[1].imshow(rollout_display, cmap="inferno", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04,
                 label="Attention Weight")

    axes[2].imshow(overlay)
    axes[2].text(5, 218, "ViT Attention Rollout", color="white",
                 fontsize=7.5, va="bottom", ha="left",
                 bbox=dict(boxstyle="round,pad=2", facecolor="#0f0f1a", alpha=0.7))

    plt.tight_layout(pad=1.5)
    return fig


def analyze_xray(
    pil_image: Image.Image,
    head_fusion: str,
    discard_ratio: float,
    threshold: float,
    explainability_method: str = "Attention Rollout",
    target_disease: str = "None (Highest Score)",
) -> tuple:
    """
    Main inference function called by Gradio.

    Returns:
        (heatmap_figure, prob_figure, diagnosis_text, status_text)
    """
    if pil_image is None:
        return None, None, "⬆ Please upload a chest X-ray image.", ""

    start_time = time.time()

    try:
        # Preprocess
        clahe_rgb, input_tensor = preprocess_uploaded_image(pil_image)

        # Inference
        model_was_training = MODEL.training
        MODEL.eval()

        with torch.no_grad():
            logits, attentions = MODEL(input_tensor.to(DEVICE), output_attentions=True)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        target_idx = np.argmax(probs)
        if target_disease != "None (Highest Score)" and target_disease in DISEASE_LABELS:
            target_idx = DISEASE_LABELS.index(target_disease)

        if explainability_method == "Attention Rollout":
            with torch.no_grad():
                _, rollout, overlay = explain_prediction(
                    model=MODEL,
                    image_tensor=input_tensor,
                    original_image=clahe_rgb,
                    device=DEVICE,
                    disease_names=DISEASE_LABELS,
                    head_fusion=head_fusion,
                    discard_ratio=discard_ratio,
                )
        else: # Grad-CAM
            MODEL.zero_grad()
            overlay = generate_gradcam_heatmap(
                model=MODEL,
                image_tensor=input_tensor.to(DEVICE),
                target_class=target_idx,
                original_image=clahe_rgb
            )
            # Create a dummy rollout to satisfy the function if Grad-CAM
            rollout = np.zeros((14, 14))

        if model_was_training:
            MODEL.train()


        elapsed = time.time() - start_time

        # Build heatmap figure
        heatmap_fig = make_heatmap_figure(clahe_rgb, rollout, overlay)

        # Build probability figure
        prob_fig = make_probability_figure(probs, threshold=threshold)

        # Build diagnosis summary text
        positives = [
            (DISEASE_LABELS[i], probs[i])
            for i in range(14)
            if probs[i] >= threshold
        ]
        positives.sort(key=lambda x: x[1], reverse=True)

        if positives:
            diag_lines = [f"### 🔴 Detected Findings (confidence ≥ {threshold:.0%})"]
            for disease, prob in positives:
                bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
                diag_lines.append(f"**{disease}**: {bar} `{prob:.1%}`")
        else:
            diag_lines = [
                f"### 🟢 No Findings Detected",
                f"All disease probabilities below threshold ({threshold:.0%}).",
                "This may indicate a normal chest X-ray.",
            ]

        diag_lines.append(f"\n---\n*Inference time: {elapsed:.2f}s · Device: {DEVICE}*")
        diag_text = "\n\n".join(diag_lines)

        status = (
            f"✅ Analysis complete in {elapsed:.2f}s | "
            f"Device: {str(DEVICE).upper()} | "
            f"{"🔥 Trained ChestViT • AUROC 0.789"}"
        )

        return heatmap_fig, prob_fig, diag_text, status

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        return None, None, f"❌ Error during analysis:\n```\n{err}\n```", "Error"


# ── Gradio Interface ──────────────────────────────────────────────────────────

def build_interface() -> gr.Blocks:
    with gr.Blocks(
        title="ChestViT -- Explainable Chest X-Ray AI",
    ) as demo:

        # Header
        gr.HTML(HEADER_HTML)

        with gr.Row():
            # ── Left Column: Input ───────────────────────────────────────────
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 📤 Upload Chest X-Ray")
                image_input = gr.Image(
                    type="pil",
                    label="Chest X-Ray (PNG/JPEG/DICOM-exported PNG)",
                    height=280,
                    sources=["upload", "clipboard"],
                )

                gr.Markdown("### ⚙️ Explainability Settings")
                with gr.Group():
                    explainability_method = gr.Radio(
                        choices=["Attention Rollout", "Grad-CAM"],
                        value="Attention Rollout",
                        label="Explainability Method",
                        info="Choose between Transformer-native Attention Rollout or Grad-CAM"
                    )
                    target_disease = gr.Dropdown(
                        choices=["None (Highest Score)"] + DISEASE_LABELS,
                        value="None (Highest Score)",
                        label="Target Disease for Grad-CAM",
                        info="Forces Grad-CAM to explain this specific disease"
                    )
                    head_fusion = gr.Radio(
                        choices=["mean", "max", "min"],
                        value="mean",
                        label="Attention Head Fusion",
                        info="[Attention Rollout] How to combine 12 attention heads into one map",
                    )
                    discard_ratio = gr.Slider(
                        minimum=0.0, maximum=0.99, value=0.9, step=0.05,
                        label="Low-Attention Discard Ratio",
                        info="[Attention Rollout] Zeroes out lowest-attention patches (noise reduction)",
                    )
                    threshold = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                        label="Prediction Threshold",
                        info="Sigmoid probability cutoff for positive prediction",
                    )

                analyze_btn = gr.Button(
                    "🔬 Analyze X-Ray",
                    variant="primary",
                    size="lg",
                )
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    show_label=True,
                    max_lines=2,
                )

                # Sample images info
                gr.Markdown(
                    """
                    > **💡 Tips**
                    > - Use frontal (PA or AP) chest X-ray images
                    > - PNG or JPEG format accepted
                    > - Best results with 1024×1024 pixel images
                    > - Works with exported DICOM screenshots
                    """
                )

            # ── Right Column: Output ─────────────────────────────────────────
            with gr.Column(scale=2, min_width=600):
                gr.Markdown("### 🔥 Attention Rollout Visualization")
                heatmap_output = gr.Plot(
                    label="Attention Rollout Analysis",
                    show_label=False,
                )

                gr.Markdown("### 📊 Disease Probability Scores")
                prob_output = gr.Plot(
                    label="Disease Probabilities",
                    show_label=False,
                )

                gr.Markdown("### 🩺 Diagnosis Summary")
                diagnosis_output = gr.Markdown(
                    value="*Upload an X-ray and click Analyze to see results.*"
                )

        # ── How It Works ──────────────────────────────────────────────────────
        with gr.Accordion("📖 How It Works", open=False):
            gr.Markdown("""
            ## Architecture

            | Component | Details |
            |---|---|
            | **Model** | ViT-Base-16 (google/vit-base-patch16-224-in21k) |
            | **Pre-training** | ImageNet-21k (14M images, 21K classes) |
            | **Fine-tuning** | NIH ChestX-ray14 (112,120 frontal X-rays) |
            | **Task** | Multi-label classification — 14 simultaneous disease predictions |
            | **Loss** | Weighted Binary Cross-Entropy (handles severe class imbalance) |
            | **Preprocessing** | CLAHE contrast enhancement → Albumentations augmentation |
            | **Explainability** | Attention Rollout (Abnar & Zuidema, 2020) |

            ## Attention Rollout Algorithm

            Standard Grad-CAM doesn't work well with pure Vision Transformers because
            ViTs don't have intermediate spatial feature maps like CNNs.

            **Attention Rollout** instead:
            1. Extracts raw attention weights from all 12 transformer layers
            2. Averages across all 12 attention heads per layer
            3. Adds an identity matrix (modeling residual/skip connections)
            4. Re-normalizes each row
            5. Multiplies all 12 matrices in sequence → propagates attention end-to-end
            6. Reads the `[CLS]` token row → shows which 14×14 patches it attends to
            7. Upsamples 14×14 → 224×224 and overlays as a heatmap

            This shows **where in the X-ray the model is looking** when it makes each prediction.

            ## NIH Dataset — 14 Disease Labels

            ```
            Atelectasis · Cardiomegaly · Effusion · Infiltration · Mass · Nodule
            Pneumonia · Pneumothorax · Consolidation · Edema · Emphysema
            Fibrosis · Pleural_Thickening · Hernia
            ```

            ## References

            - Wang et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks*. CVPR.
            - Dosovitskiy et al. (2021). *An Image is Worth 16x16 Words*. ICLR.
            - Abnar & Zuidema (2020). *Quantifying Attention Flow in Transformers*. arXiv:2005.00928.
            """)

        gr.HTML(FOOTER_HTML)

        # ── Event Binding ─────────────────────────────────────────────────────
        analyze_btn.click(
            fn=analyze_xray,
            inputs=[image_input, head_fusion, discard_ratio, threshold, explainability_method, target_disease],
            outputs=[heatmap_output, prob_output, diagnosis_output, status_text],
            api_name="analyze",
        )

        # Also trigger on image upload (optional — comment out to disable auto-run)
        # image_input.change(
        #     fn=analyze_xray,
        #     inputs=[image_input, head_fusion, discard_ratio, threshold],
        #     outputs=[heatmap_output, prob_output, diagnosis_output, status_text],
        # )

    return demo


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_port=cfg.inference.gradio_port,
        share=cfg.inference.gradio_share,
        show_error=True,
        inbrowser=True,
        theme=gr.themes.Base(
            primary_hue="indigo",
            secondary_hue="cyan",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif"],
        ),
        css=CUSTOM_CSS,
    )
