"""
data/download.py
----------------
Downloads the NIH ChestX-ray14 dataset from Kaggle using the Kaggle API.

SETUP (one-time):
  1. Create a Kaggle account at https://www.kaggle.com
  2. Go to: Account → Settings → Create New API Token → downloads kaggle.json
  3. Place kaggle.json at:
       Windows : C:\\Users\\<YourName>\\.kaggle\\kaggle.json
       Linux   : ~/.kaggle/kaggle.json
  4. Run:  python data/download.py

Dataset: nih-chest-xrays/data
  - 112,120 frontal chest X-ray images (PNG, 1024×1024)
  - Data_Entry_2017.csv  — image metadata + labels
  - train_val_list.txt   — official train+val split
  - test_list.txt        — official test split
  Total size: ~42 GB (compressed ~11 GB — the script downloads zip chunks)
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
from typing import Optional

# ── Rich for pretty console output ───────────────────────────────────────────
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich.panel import Panel

    console = Console()
except ImportError:
    import builtins

    class _FallbackConsole:
        def print(self, *args, **kwargs):
            builtins.print(*args)
        def rule(self, *args, **kwargs):
            builtins.print("─" * 60)

    console = _FallbackConsole()


# ── Constants ─────────────────────────────────────────────────────────────────
KAGGLE_DATASET = "nih-chest-xrays/data"
DEFAULT_RAW_DIR = Path("./data/raw")

REQUIRED_FILES = [
    "Data_Entry_2017.csv",
    "train_val_list.txt",
    "test_list.txt",
]


def _check_kaggle_credentials() -> bool:
    """Verify kaggle.json exists and is readable."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    if not kaggle_json.exists():
        console.print(
            "[bold red]✗ kaggle.json not found![/bold red]\n"
            f"  Expected location: {kaggle_json}\n\n"
            "  Steps to fix:\n"
            "  1. Visit https://www.kaggle.com → Account → Settings\n"
            "  2. Click 'Create New API Token' → downloads kaggle.json\n"
            f"  3. Move it to: {kaggle_dir}\n"
            "  4. Re-run this script.",
            style="red",
        )
        return False
    # Ensure correct permissions (important on Linux/macOS)
    if os.name != "nt":
        kaggle_json.chmod(0o600)
    return True


def _verify_images_dir(raw_dir: Path) -> bool:
    """
    Check if images are already extracted to avoid re-downloading.
    We consider extraction complete if >1000 .png files exist.
    """
    images_dir = raw_dir / "images"
    if not images_dir.exists():
        return False
    png_count = sum(1 for _ in images_dir.glob("*.png"))
    console.print(f"  Found {png_count:,} existing PNG files in {images_dir}")
    return png_count > 1000


def download_dataset(
    raw_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Download and extract NIH ChestX-ray14 from Kaggle.

    Args:
        raw_dir: Directory to extract data into.
        force:   Re-download even if files already exist.

    Returns:
        Path to the extracted raw data directory.
    """
    raw_dir = Path(raw_dir or DEFAULT_RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Check credentials ──────────────────────────────────────────────────
    console.print(Panel("[bold cyan]NIH ChestX-ray14 — Dataset Download[/bold cyan]",
                        subtitle="Kaggle API"))
    if not _check_kaggle_credentials():
        sys.exit(1)

    # ── 2. Check if already downloaded ───────────────────────────────────────
    required_present = all((raw_dir / f).exists() for f in REQUIRED_FILES)
    images_present = _verify_images_dir(raw_dir)

    if required_present and images_present and not force:
        console.print("[bold green]✓ Dataset already downloaded and extracted.[/bold green]")
        console.print(f"  Data root: {raw_dir.resolve()}")
        return raw_dir

    # ── 3. Import kaggle (after credentials check) ────────────────────────────
    try:
        import kaggle  # noqa: F401 — triggers credential load
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
    except ImportError:
        console.print("[red]✗ kaggle package not installed. Run: pip install kaggle[/red]")
        sys.exit(1)

    api = KaggleApiExtended()
    api.authenticate()

    # ── 4. Download dataset zip(s) ────────────────────────────────────────────
    console.print(f"\n[yellow]⬇ Downloading '{KAGGLE_DATASET}' → {raw_dir.resolve()}[/yellow]")
    console.print("  ⚠ This dataset is ~42 GB. Download time depends on your connection.")
    console.print("  ⚠ The Kaggle API downloads each image zip file sequentially.\n")

    api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(raw_dir),
        unzip=False,     # We handle extraction ourselves for better control
        quiet=False,
    )
    console.print("[green]✓ Download complete.[/green]")

    # ── 5. Extract all zip files ──────────────────────────────────────────────
    images_dir = raw_dir / "images"
    images_dir.mkdir(exist_ok=True)

    zip_files = sorted(raw_dir.glob("*.zip")) + sorted(raw_dir.glob("images_*.tar.gz"))
    console.print(f"\n[yellow]📦 Extracting {len(zip_files)} archive(s)...[/yellow]")

    for zip_path in zip_files:
        console.print(f"  Extracting: {zip_path.name}")
        if zip_path.suffix == ".zip":
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Extract images to images_dir, metadata to raw_dir
                for member in zf.namelist():
                    if member.endswith(".png"):
                        # Extract flat into images_dir
                        target = images_dir / Path(member).name
                        if not target.exists():
                            with zf.open(member) as src, open(target, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                    elif not Path(member).is_dir():
                        zf.extract(member, raw_dir)
        console.print(f"    ✓ {zip_path.name} extracted.")

    # ── 6. Verify essential files ─────────────────────────────────────────────
    console.rule("Verification")
    all_ok = True
    for fname in REQUIRED_FILES:
        fpath = raw_dir / fname
        status = "[green]✓[/green]" if fpath.exists() else "[red]✗ MISSING[/red]"
        console.print(f"  {status} {fname}")
        if not fpath.exists():
            all_ok = False

    png_count = sum(1 for _ in images_dir.glob("*.png"))
    console.print(f"  [cyan]PNG images found: {png_count:,}[/cyan]")

    if all_ok and png_count > 1000:
        console.print("\n[bold green]🎉 Dataset ready![/bold green]")
        console.print(f"  Data root : {raw_dir.resolve()}")
        console.print(f"  Images    : {images_dir.resolve()}")
    else:
        console.print("\n[bold red]⚠ Some files may be missing. Check the output above.[/bold red]")

    return raw_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download NIH ChestX-ray14 from Kaggle")
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_RAW_DIR),
        help="Directory to save the dataset (default: ./data/raw)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if dataset already exists",
    )
    args = parser.parse_args()
    download_dataset(raw_dir=Path(args.output_dir), force=args.force)
