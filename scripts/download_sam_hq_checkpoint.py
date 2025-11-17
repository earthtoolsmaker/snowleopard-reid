"""Download SAM HQ model checkpoint from HuggingFace.

Usage:
    uv run python scripts/download_sam_hq_checkpoint.py [model_type]

Arguments:
    model_type: vit_b (default), vit_l, or vit_h

Examples:
    uv run python scripts/download_sam_hq_checkpoint.py           # Downloads vit_b
    uv run python scripts/download_sam_hq_checkpoint.py vit_l     # Downloads vit_l
    uv run python scripts/download_sam_hq_checkpoint.py vit_h     # Downloads vit_h
"""

import argparse
import sys
import urllib.request
from pathlib import Path


def download_with_progress(url: str, destination: Path) -> None:
    """Download file with progress bar."""

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            print(
                f"\rDownloading: {mb_downloaded:.1f}MB / {mb_total:.1f}MB ({percent:.1f}%)",
                end="",
                flush=True,
            )

    print(f"Downloading from: {url}")
    print(f"Saving to: {destination}")
    print()

    urllib.request.urlretrieve(url, destination, progress_hook)
    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download SAM HQ model checkpoint from HuggingFace"
    )
    parser.add_argument(
        "model_type",
        nargs="?",
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="Model type to download (default: vit_b)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )

    args = parser.parse_args()

    # Model info
    model_sizes = {
        "vit_b": "~379MB",
        "vit_l": "~1.2GB",
        "vit_h": "~2.4GB",
    }

    model_type = args.model_type
    size = model_sizes[model_type]

    # Create directory
    models_dir = Path("data/04_models/SAM_HQ")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Check if model exists
    model_path = models_dir / f"sam_hq_{model_type}.pth"
    if model_path.exists() and not args.force:
        file_size = model_path.stat().st_size / 1024 / 1024  # MB
        print(f"Model already exists: {model_path}")
        print(f"File size: {file_size:.1f}MB")
        print()
        response = input("Re-download? (y/N): ").strip().lower()
        if response != "y":
            print("Skipping download.")
            return 0
        print("Removing existing model...")
        model_path.unlink()

    # Download URL
    url = f"https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_{model_type}.pth"

    print("=" * 60)
    print("Downloading SAM HQ Model")
    print("=" * 60)
    print(f"Model type: {model_type}")
    print(f"Expected size: {size}")
    print("=" * 60)
    print()

    try:
        download_with_progress(url, model_path)

        file_size = model_path.stat().st_size / 1024 / 1024  # MB
        print("=" * 60)
        print("Download complete!")
        print(f"Model saved to: {model_path}")
        print(f"File size: {file_size:.1f}MB")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nError downloading model: {e}", file=sys.stderr)
        if model_path.exists():
            model_path.unlink()
        return 1


if __name__ == "__main__":
    sys.exit(main())
