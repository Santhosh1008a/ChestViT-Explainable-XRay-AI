# Multi-Task Vision Transformer for Chest X-Ray Disease Classification & Explainability

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface)
![Gradio](https://img.shields.io/badge/Gradio-4.x-FF7C00?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square)

**ViT-Base-16 · 14-Disease Multi-Label Classification · Attention Rollout XAI**

*Fine-tuned on NIH ChestX-ray14 (112,120 frontal chest X-rays)*

</div>

---

## 🎯 Overview

This project implements an **explainable medical AI system** for automated chest X-ray analysis. Unlike standard classification models, this system simultaneously:

1. **Predicts 14 diseases** in parallel (multi-label, not multi-class)
2. **Shows WHERE** in the X-ray the model is looking via Attention Rollout
3. **Compares against** published NIH baselines (AUC-ROC per class)
4. **Serves a live demo** via a Gradio dashboard

The core insight: Vision Transformers (ViTs) divide images into 16×16 patches and route information through 12 attention layers. **Attention Rollout** traces this information flow back to the input patches — telling us exactly which lung regions drove each disease prediction.

---

## 🏗 Architecture

```
Chest X-Ray Input (PNG, 1024×1024)
           │
           ▼
   ┌───────────────────────┐
   │  CLAHE Preprocessing  │   ← Contrast Limited Adaptive Histogram Equalization
   │  + Albumentations     │   ← Radiologically-realistic augmentation
   └───────────────────────┘
           │ 224×224×3
           ▼
   ┌───────────────────────────────────────────┐
   │          ViT-Base-16                      │
   │  (google/vit-base-patch16-224-in21k)      │
   │                                           │
   │  ┌─────────────────────────────────────┐  │
   │  │  196 Patches (14×14 grid, 16px/ea) │  │
   │  │  + [CLS] token = 197 total tokens  │  │
   │  └─────────────────────────────────────┘  │
   │              ↓                            │
   │  12 Transformer Layers                    │
   │  (12 heads × 64 dim = 768 hidden dim)    │
   │              ↓                            │
   │  [CLS] token → Dropout → Linear(768→14)  │
   └───────────────────────────────────────────┘
           │
           ├─── Logits → Sigmoid → 14 disease probabilities
           │
           └─── Attention weights (12 layers × 12 heads)
                         │
                         ▼
              Attention Rollout Algorithm
              (14×14 patch attention map)
                         │
                         ▼
              224×224 heatmap overlay on X-ray
```

---

## 📊 Results

| Disease | ViT AUC | NIH Baseline | Δ AUC |
|---|---|---|---|
| Atelectasis | — | 0.7003 | — |
| Cardiomegaly | — | 0.8100 | — |
| Effusion | — | 0.7585 | — |
| Infiltration | — | 0.6614 | — |
| Mass | — | 0.6933 | — |
| Nodule | — | 0.6689 | — |
| Pneumonia | — | 0.6580 | — |
| Pneumothorax | — | 0.7993 | — |
| Consolidation | — | 0.7032 | — |
| Edema | — | 0.8052 | — |
| Emphysema | — | 0.8330 | — |
| Fibrosis | — | 0.7859 | — |
| Pleural_Thickening | — | 0.6835 | — |
| Hernia | — | 0.8717 | — |
| **MACRO AVERAGE** | — | **0.7523** | — |

*Results will populate after training. NIH baseline from Wang et al. (2017).*

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# First: set up Kaggle API credentials
# 1. Go to https://www.kaggle.com → Account → Settings → Create New API Token
# 2. Place kaggle.json at: C:\Users\<YourName>\.kaggle\kaggle.json

# Then download NIH ChestX-ray14 (~42 GB)
python data/download.py
```

### 3. Run Unit Tests

```bash
python -m pytest tests/ -v
```

### 4. Train the Model

```bash
# Full training (5 epochs, ~8-12 hours on RTX 3050)
python training/train.py

# Quick smoke test (20% of data)
# Edit config/config.yaml → dataset.train_fraction: 0.2
python training/train.py
```

Monitor training in real-time:
```bash
mlflow ui --backend-store-uri ./experiments/mlflow
# Open http://localhost:5000
```

### 5. Launch Demo

```bash
# With trained model
python app/gradio_app.py

# DEMO MODE (random weights, for UI preview only)
set DEMO_MODE=1   # Windows
python app/gradio_app.py
```

---

## 📁 Project Structure

```
.
├── config/
│   └── config.yaml              # All hyperparameters and paths
├── data/
│   ├── download.py              # Kaggle API dataset download
│   ├── preprocessing.py         # CLAHE + Albumentations pipeline
│   ├── dataset.py               # ChestXrayDataset + DataLoaders
│   └── raw/                     # Downloaded dataset (not in git)
├── models/
│   └── vit_model.py             # ViT-Base-16 with multi-label head
├── explainability/
│   └── attention_rollout.py     # Attention Rollout algorithm
├── training/
│   ├── losses.py                # Weighted BCE + Focal Loss
│   ├── train.py                 # Training loop (mixed precision, MLflow)
│   └── evaluate.py              # AUC-ROC per class, ROC plots
├── app/
│   └── gradio_app.py            # Gradio dashboard
├── tests/
│   └── test_modules.py          # Unit tests (no dataset required)
├── checkpoints/                 # Saved model weights (not in git)
├── results/                     # ROC curves, metrics CSV
├── experiments/
│   └── mlflow/                  # MLflow tracking database
├── config_loader.py             # YAML config loader
└── requirements.txt
```

---

## ⚙️ Configuration

All settings are in [`config/config.yaml`](config/config.yaml). Key RTX 3050 settings:

```yaml
training:
  batch_size: 8                    # Fits in 4 GB VRAM
  gradient_accumulation_steps: 4   # Effective batch = 32
  mixed_precision: true            # fp16 — mandatory for 4 GB VRAM
  num_epochs: 5

model:
  name: "google/vit-base-patch16-224-in21k"
  gradient_checkpointing: true     # Saves ~30% VRAM

dataset:
  train_fraction: 1.0              # Set 0.2 for quick smoke test
```

---

## 🔥 Attention Rollout: Why Not Grad-CAM?

| Method | Grad-CAM | Attention Rollout |
|---|---|---|
| **Designed for** | CNNs | Transformers |
| **Spatial resolution** | Depends on last conv layer | 14×14 patch grid |
| **Accounts for skip connections** | No | Yes (identity matrix) |
| **Computational cost** | Requires backward pass | Forward pass only |
| **ViT-specific** | No | Yes |

Attention Rollout (Abnar & Zuidema, 2020) is mathematically derived from the transformer's own attention mechanism, making it the correct tool for ViT explainability.

---

## 🏥 Clinical Context

> ⚠️ **This is a research/educational project, NOT a medical device.**
> Results should not be used for clinical diagnosis without radiologist review.

The NIH ChestX-ray14 dataset has known limitations (Rajpurkar et al., 2018, and others). AUC-ROC is the clinically relevant metric because:
- Accuracy is misleading with class imbalance (>53% "No Finding")
- AUC measures discriminative ability across all thresholds
- Radiologists can set their own confidence threshold per clinical context

---

## 📚 References

1. Wang, X. et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks.* CVPR.
2. Dosovitskiy, A. et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR.
3. Abnar, S. & Zuidema, W. (2020). *Quantifying Attention Flow in Transformers.* arXiv:2005.00928.
4. Rajpurkar, P. et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.* arXiv:1711.05225.

---

## 🛠 Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| PyTorch | 2.1+ | Training framework |
| HuggingFace Transformers | 4.37+ | ViT-Base-16 backbone |
| OpenCV | 4.9+ | CLAHE preprocessing |
| Albumentations | 1.3+ | Image augmentation |
| scikit-learn | 1.4+ | AUC-ROC metrics |
| MLflow | 2.10+ | Experiment tracking |
| Gradio | 4.x | Demo dashboard |
| Kaggle API | 1.6+ | Dataset download |
