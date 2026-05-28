"""
demo_launch.py
--------------
One-command launcher for the ChestViT Gradio demo.
Handles Windows encoding, DEMO_MODE, and opens browser automatically.

Usage:
  python demo_launch.py              # Auto-detect: use trained model if exists
  python demo_launch.py --demo       # Force DEMO_MODE (random weights, works instantly)
  python demo_launch.py --port 7861  # Custom port
"""

import os
import sys
import argparse
from pathlib import Path

# MUST be first — fix Windows console encoding before any other imports
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description="Launch ChestViT Gradio Demo")
    parser.add_argument("--demo", action="store_true",
                        help="Force DEMO_MODE (random weights, no checkpoint needed)")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port for Gradio server (default: 7860)")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    args = parser.parse_args()

    # Check for trained checkpoint
    ckpt = Path("checkpoints/best_model.pt")
    if not ckpt.exists() and not args.demo:
        print("\n  No trained checkpoint found at checkpoints/best_model.pt")
        print("  Automatically switching to DEMO_MODE (random weights).")
        print("  UI and attention rollout will be fully functional.")
        print("  To get real predictions: run python training/train.py first.\n")
        args.demo = True

    if args.demo:
        os.environ["DEMO_MODE"] = "1"
        print("\n  [DEMO MODE] Starting with random weights.")
        print("  Attention rollout maps will still show real spatial patterns.\n")
    else:
        print(f"\n  Loading trained model from: {ckpt}")

    # Override config port if specified
    if args.port != 7860:
        # Temporarily patch config
        import yaml
        cfg_path = Path("config/config.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg["inference"]["gradio_port"] = args.port
        cfg["inference"]["gradio_share"] = args.share
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print("  Starting Gradio server...")
    print(f"  URL: http://localhost:{args.port}")
    print("  Press Ctrl+C to stop.\n")

    # Import and run
    from app.gradio_app import build_interface
    from config_loader import load_config

    cfg = load_config()
    demo = build_interface()
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
