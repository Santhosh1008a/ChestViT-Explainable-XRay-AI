"""
demo_setup.py
--------------
Portfolio demo setup — NO 42 GB download needed.

This script sets up a fully working demo using ONE of two strategies:

STRATEGY A (Recommended for portfolio):
  Downloads the NIH sample subset from Kaggle (~50 MB, ~1000 images)
  Trains for 3 epochs on this tiny subset (~10-15 minutes on RTX 3050)
  Result: Real trained model with real AUC scores to show

STRATEGY B (Instant demo, no training):
  Downloads 10 real chest X-ray images from public sources
  Runs Gradio in DEMO_MODE — UI is fully functional, predictions are random
  Good for showing the interface before training

Usage:
  python demo_setup.py --strategy A    # Download sample + train
  python demo_setup.py --strategy B    # Just get demo images
  python demo_setup.py --strategy B --skip-download  # Use your own images
"""

import os
import sys
import shutil
import urllib.request
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "data" / "demo_images"

# ── Public domain chest X-ray images from NIH / OpenI ────────────────────────
# These are from the NIH Clinical Center public image sharing project
# and the Indiana University chest X-ray dataset (public domain)
SAMPLE_XRAY_URLS = [
    # From NIH open-access image repository
    ("https://openi.nlm.nih.gov/imgs/512/0/0/0_IM-0001-1001.png",  "sample_01.png"),
    ("https://openi.nlm.nih.gov/imgs/512/0/1/1_IM-0003-1001.png",  "sample_02.png"),
    ("https://openi.nlm.nih.gov/imgs/512/0/2/2_IM-0004-1001.png",  "sample_03.png"),
    ("https://openi.nlm.nih.gov/imgs/512/0/3/3_IM-0005-1001.png",  "sample_04.png"),
    ("https://openi.nlm.nih.gov/imgs/512/0/4/4_IM-0006-1001.png",  "sample_05.png"),
]


def strategy_b_download_samples():
    """Download a handful of real public-domain chest X-rays for demo."""
    SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Strategy B] Downloading sample chest X-ray images...")
    print("  Source: OpenI / NIH public image repository\n")

    downloaded = 0
    for url, fname in SAMPLE_XRAY_URLS:
        dest = SAMPLE_IMAGES_DIR / fname
        if dest.exists():
            print(f"  [skip] {fname} already exists")
            downloaded += 1
            continue
        try:
            print(f"  Downloading {fname}...", end=" ", flush=True)
            urllib.request.urlretrieve(url, dest)
            print("OK")
            downloaded += 1
        except Exception as e:
            print(f"FAILED ({e})")

    if downloaded == 0:
        print("\n  Could not download sample images (network issue?).")
        print("  Manual option: Download any chest X-ray PNG and place it in:")
        print(f"  {SAMPLE_IMAGES_DIR}")
    else:
        print(f"\n  {downloaded} sample images ready at: {SAMPLE_IMAGES_DIR}")

    return downloaded > 0


def strategy_a_kaggle_sample():
    """Download the small NIH sample dataset from Kaggle (~50 MB)."""
    # Check kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n  [ERROR] kaggle.json not found.")
        print("  To get it: kaggle.com -> Account -> Settings -> Create API Token")
        print(f"  Save to: {kaggle_json}")
        print("\n  Falling back to Strategy B (download sample images)...")
        return strategy_b_download_samples()

    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()
    except Exception as e:
        print(f"  [ERROR] Kaggle auth failed: {e}")
        return strategy_b_download_samples()

    sample_dir = PROJECT_ROOT / "data" / "raw"
    sample_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Strategy A] Downloading NIH ChestX-ray14 SAMPLE subset from Kaggle...")
    print("  Dataset: nih-chest-xrays/sample (~50 MB, ~1000 images)")
    print("  This is the small sample version — NOT the 42 GB full dataset.\n")

    import zipfile
    api.dataset_download_files(
        "nih-chest-xrays/sample",
        path=str(sample_dir),
        unzip=False,
        quiet=False,
    )

    # Extract
    images_dir = sample_dir / "images"
    images_dir.mkdir(exist_ok=True)
    for zf_path in sample_dir.glob("*.zip"):
        print(f"\n  Extracting {zf_path.name}...")
        with zipfile.ZipFile(zf_path) as zf:
            for member in zf.namelist():
                if member.endswith(".png"):
                    target = images_dir / Path(member).name
                    if not target.exists():
                        with zf.open(member) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
        print(f"  Done.")

    png_count = sum(1 for _ in images_dir.glob("*.png"))
    print(f"\n  {png_count} images extracted to: {images_dir}")
    return png_count > 0


def patch_config_for_sample():
    """Update config.yaml to point to sample data and use lightweight settings."""
    import yaml

    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Point to sample images
    cfg["paths"]["images_dir"] = "./data/raw/images"
    cfg["paths"]["labels_csv"] = "./data/raw/sample/Data_Entry_2017.csv"
    cfg["paths"]["train_list"] = "./data/raw/sample/train_val_list.txt"
    cfg["paths"]["test_list"]  = "./data/raw/sample/test_list.txt"

    # Ultra-lightweight training for demo
    cfg["training"]["num_epochs"] = 3
    cfg["training"]["batch_size"] = 8
    cfg["dataset"]["train_fraction"] = 1.0  # Use all ~800 sample train images

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print("\n  config.yaml updated for sample training.")


def print_next_steps(strategy: str):
    print("\n" + "=" * 60)
    print("  SETUP COMPLETE — What to do next:")
    print("=" * 60)

    if strategy == "A":
        print("""
  1. TRAIN on sample data (~10-15 min on RTX 3050):
       python training/train.py

  2. LAUNCH the Gradio demo:
       python demo_launch.py

  3. UPLOAD one of the sample images in data/raw/images/
     to see real predictions + attention rollout heatmap.

  4. TAKE SCREENSHOTS for your portfolio!

  TIP: After training, your results table will show
       real AUC-ROC scores per disease class.
""")
    else:
        print("""
  1. LAUNCH the Gradio demo (runs with random weights):
       python demo_launch.py

  2. UPLOAD any chest X-ray image (PNG/JPEG).
     Sample images are in: data/demo_images/
     Or find chest X-rays on Google Images / Radiopaedia.

  3. SCREENSHOT the UI for your portfolio.
     The attention heatmap + probability chart will
     look exactly like the real trained version.

  NOTE: Predictions will be random (untrained model).
        Train on sample data (Strategy A) for real scores.
""")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Portfolio demo setup")
    parser.add_argument("--strategy", choices=["A", "B"], default="B",
                        help="A=download sample+train, B=just demo images (default)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip image download (you have your own X-ray images)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ChestViT Portfolio Demo Setup")
    print("=" * 60)

    if args.skip_download:
        print(f"\n  Skipping download. Place your X-ray images in:")
        print(f"  {SAMPLE_IMAGES_DIR}")
        SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    elif args.strategy == "A":
        strategy_a_kaggle_sample()
    else:
        strategy_b_download_samples()

    print_next_steps(args.strategy)
