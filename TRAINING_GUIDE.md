# ChestViT Training Guide (Stage 2)

This guide provides instructions on how to properly fine-tune ChestViT on a GPU environment using the Hugging Face `BahaaEldin0/NIH-Chest-Xray-14` streaming dataset.

## Recommended Settings for Stage 2
For a comprehensive but efficient training phase, the following settings are recommended for fine-tuning on 20,000 images:

- **Dataset Subset:** 20,000 streamed images (`train_take_limit: 20000` in `config/config.yaml`)
- **Validation Subset:** 2,000 streamed images
- **Epochs:** 10 to 20
- **Batch Size:** 8 to 32 (Depends strictly on VRAM. A 4 GB GPU typically fits batch size 8 with mixed precision and gradient checkpointing enabled).
- **Gradient Accumulation:** 4 (Effective batch size = 32)
- **Mixed Precision:** True (fp16)
- **Learning Rate:** 2.0e-5 (AdamW)
- **Explainability Validation:** Grad-CAM & Attention Rollout

## Environment Setups

### 1. Local RTX 3050 (or similar 4-8 GB VRAM GPU)
If you have a local NVIDIA GPU with at least 4 GB VRAM:
1. Ensure CUDA and cuDNN are installed on your OS.
2. Create your Python environment and install the requirements:
   ```bash
   pip install -r requirements.txt
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
3. Update `config/config.yaml`:
   - `training.batch_size`: 8
   - `training.mixed_precision`: true
   - `model.gradient_checkpointing`: true
4. Run the training script:
   ```bash
   python training/train.py
   ```
5. Check results in the MLflow UI:
   ```bash
   mlflow ui --backend-store-uri ./experiments/mlflow
   ```

### 2. Google Colab (T4 / L4 / A100)
Google Colab provides an excellent environment for this project, offering 15-40 GB VRAM.
1. Upload this repository to Google Drive or clone it directly into a Colab cell.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Install dependencies:
   ```python
   !pip install datasets huggingface_hub pillow albumentations pytorch-grad-cam sentence-transformers faiss-cpu timm mlflow
   ```
4. Authenticate with Hugging Face (optional but recommended for faster downloads):
   ```python
   from huggingface_hub import login
   login("YOUR_HF_TOKEN")
   ```
5. Update `config/config.yaml` to utilize more VRAM:
   - `training.batch_size`: 32 (or higher depending on the specific GPU)
   - `training.gradient_accumulation_steps`: 1
6. Execute training:
   ```python
   !python training/train.py
   ```

### 3. Kaggle Notebooks (P100 / T4x2)
Kaggle Notebooks provide 16 GB VRAM GPUs and integrate naturally with Hugging Face and MLflow.
1. Create a new Kaggle Notebook and attach the source code.
2. Set the Accelerator to **GPU P100** or **GPU T4x2**.
3. Install missing packages:
   ```bash
   !pip install datasets huggingface_hub pytorch-grad-cam sentence-transformers faiss-cpu timm mlflow
   ```
4. Execute training using the standard command:
   ```bash
   !python training/train.py
   ```

## Interpreting Results
- MLflow will automatically log metrics including AUROC, F1-score, loss, precision, and recall.
- Once training completes (or upon reaching early stopping criteria), evaluation metrics CSV, ROC Curves, and Confusion Matrices will be automatically exported to the `results/` folder.
- Always use the `results/best_model.pt` checkpoint for deployment via `app/gradio_app.py`.
