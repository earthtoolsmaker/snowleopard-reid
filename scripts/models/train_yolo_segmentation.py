"""Script to train a YOLO segmentation model on snow leopard images.

This script fine-tunes YOLO segmentation models for snow leopard detection and segmentation.
It supports training from pretrained checkpoints or custom models, with configurable
hyperparameters for experimentation and optimization.

This script:
1. Loads pretrained YOLO segmentation model (yolo11n-seg, yolo11s-seg, etc.)
2. Trains on YOLO dataset with train/val/test splits
3. Auto-detects and uses GPU when available
4. Saves model checkpoints (best.pt and last.pt)
5. Generates training visualizations (loss curves, confusion matrix, predictions)
6. Logs metrics and validation results during training

Usage:
    python scripts/models/train_yolo_segmentation.py --data-yaml <path> --output-dir <path> [options]

Example:
    python scripts/models/train_yolo_segmentation.py \
        --data-yaml "./data/03_model_input/yolo/segmentation/data.yaml" \
        --output-dir "./data/04_models/yolo/segmentation/baseline" \
        --model yolo11n-seg \
        --epochs 10 \
        --batch-size 32 \
        --imgsz 640

Notes:
    - Model checkpoints saved to output_dir/weights/
    - Best model selected based on validation mAP50-95
    - Use larger image sizes (1024, 1280) for better small object detection
    - Batch size should be adjusted based on GPU memory
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def get_device() -> str:
    """
    Determine the best available device (CUDA GPU or CPU).

    Returns:
        str: Device identifier for YOLO ('cuda:0' or 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda:0"
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        device = "cpu"
        logging.info("Using CPU (no GPU available)")

    return device


def validate_model_name(model_name: str) -> None:
    """
    Validate the YOLO model name.

    Args:
        model_name: Model name to validate

    Raises:
        ValueError: If model name is invalid
    """
    valid_models = [
        "yolo11n-seg",
        "yolo11s-seg",
        "yolo11m-seg",
        "yolo11l-seg",
        "yolo11x-seg",
        "yolo12n-seg",
        "yolo12s-seg",
        "yolo12m-seg",
        "yolo12l-seg",
        "yolo12x-seg",
    ]

    if model_name not in valid_models:
        raise ValueError(
            f"Invalid model name: {model_name}. Must be one of: {', '.join(valid_models)}"
        )


def train_model(
    data_yaml: Path,
    output_dir: Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    device: str,
    fliplr: float = 0.5,
    scale: float = 0.5,
    translate: float = 0.1,
    lr0: float = 0.01,
    lrf: float = 0.01,
    warmup_epochs: float = 3.0,
    patience: int = 50,
) -> dict[str, Any]:
    """
    Train a YOLO segmentation model.

    Args:
        data_yaml: Path to YOLO data.yaml configuration file
        output_dir: Directory to save training outputs
        model_name: YOLO model variant (e.g., 'yolo12n-seg')
        epochs: Number of training epochs
        batch_size: Training batch size
        imgsz: Input image size
        device: Device to train on ('cuda:0' or 'cpu')
        fliplr: Probability of horizontal flip augmentation (0.0-1.0)
        scale: Image scale augmentation factor (+/- gain, 0.0-1.0)
        translate: Image translation augmentation fraction (+/- fraction, 0.0-1.0)
        lr0: Initial learning rate (0.0-1.0)
        lrf: Final learning rate factor (final_lr = lr0 * lrf, 0.0-1.0)
        warmup_epochs: Number of warmup epochs (0.0+)
        patience: Early stopping patience in epochs

    Returns:
        Dictionary containing training results and metrics
    """
    logging.info(f"Loading model: {model_name}")

    # Load pretrained model (ultralytics downloads weights automatically)
    # Pass just the model name, YOLO will download the .pt file if needed
    model = YOLO(model_name)

    logging.info("Starting training...")

    # Train the model
    # See: https://docs.ultralytics.com/modes/train/
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=str(output_dir.parent),  # Parent directory
        name=output_dir.name,  # Run name (will create output_dir)
        exist_ok=True,  # Allow overwriting existing run
        pretrained=True,  # Use pretrained weights
        verbose=True,  # Show detailed training progress
        plots=True,  # Generate training plots
        save=True,  # Save checkpoints
        save_period=-1,  # Only save last and best (don't save every N epochs)
        # Learning rate parameters
        lr0=lr0,  # Initial learning rate
        lrf=lrf,  # Final learning rate factor
        warmup_epochs=warmup_epochs,  # Warmup duration
        patience=patience,  # Early stopping patience
        # Data augmentation parameters
        fliplr=fliplr,  # Horizontal flip probability
        scale=scale,  # Image scale (+/- gain)
        translate=translate,  # Image translation (+/- fraction)
    )

    logging.info("Training completed!")

    return results


def print_training_summary(output_dir: Path) -> None:
    """
    Print a summary of training results.

    Args:
        output_dir: Directory containing training outputs
    """
    weights_dir = output_dir / "weights"
    best_weights = weights_dir / "best.pt"
    last_weights = weights_dir / "last.pt"

    logging.info("=" * 70)
    logging.info("Training Summary")
    logging.info("=" * 70)

    if best_weights.exists():
        logging.info(f"Best weights: {best_weights}")
    if last_weights.exists():
        logging.info(f"Last weights: {last_weights}")

    # List output files
    output_files = [
        "results.png",
        "results.csv",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "PR_curve.png",
    ]

    logging.info("\nGenerated outputs:")
    for filename in output_files:
        filepath = output_dir / filename
        if filepath.exists():
            logging.info(f"  - {filepath}")

    logging.info("=" * 70)


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Train a YOLO segmentation model on snow leopard images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s --data-yaml "./data/03_model_input/yolo/segmentation/data.yaml" \\
           --output-dir "./data/04_models/yolo/segmentation/baseline" \\
           --model yolo12n-seg \\
           --epochs 10 \\
           --batch-size 32 \\
           --imgsz 640

Available models:
  YOLO11: yolo11n-seg, yolo11s-seg, yolo11m-seg, yolo11l-seg, yolo11x-seg
  YOLO12: yolo12n-seg, yolo12s-seg, yolo12m-seg, yolo12l-seg, yolo12x-seg

  Model sizes (from smallest to largest):
    n (nano)   - Fastest, lowest accuracy
    s (small)  - Good balance
    m (medium) - Better accuracy
    l (large)  - High accuracy
    x (xlarge) - Best accuracy, slowest
        """,
    )

    parser.add_argument(
        "--data-yaml",
        type=Path,
        required=True,
        help="Path to YOLO data.yaml configuration file",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for training results and model weights",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo12n-seg",
        help="YOLO model variant (default: yolo12n-seg)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32, use -1 for auto)",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )

    parser.add_argument(
        "--fliplr",
        type=float,
        default=0.5,
        help="Horizontal flip probability for augmentation (default: 0.5, range: 0.0-1.0)",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Image scale augmentation factor (default: 0.5, range: 0.0-1.0)",
    )

    parser.add_argument(
        "--translate",
        type=float,
        default=0.1,
        help="Image translation augmentation fraction (default: 0.1, range: 0.0-1.0)",
    )

    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01)",
    )

    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate factor (final_lr = lr0 * lrf, default: 0.01)",
    )

    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=3.0,
        help="Number of warmup epochs (default: 3.0)",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience in epochs (default: 50)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Validate inputs
    if not args.data_yaml.exists():
        logging.error(f"Data YAML not found: {args.data_yaml}")
        sys.exit(1)

    if not args.data_yaml.is_file():
        logging.error(f"Data YAML path is not a file: {args.data_yaml}")
        sys.exit(1)

    if args.epochs <= 0:
        logging.error(f"Epochs must be positive, got: {args.epochs}")
        sys.exit(1)

    if args.imgsz <= 0:
        logging.error(f"Image size must be positive, got: {args.imgsz}")
        sys.exit(1)

    try:
        validate_model_name(args.model)
    except ValueError as e:
        logging.error(str(e))
        sys.exit(1)

    logging.info("=" * 70)
    logging.info("YOLO Segmentation Training Pipeline")
    logging.info("=" * 70)
    logging.info(f"Data YAML: {args.data_yaml}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Image size: {args.imgsz}")
    logging.info(f"Learning Rate - lr0: {args.lr0}")
    logging.info(f"Learning Rate - lrf: {args.lrf}")
    logging.info(f"Learning Rate - warmup epochs: {args.warmup_epochs}")
    logging.info(f"Early Stopping - patience: {args.patience}")
    logging.info(f"Augmentation - Flip LR: {args.fliplr}")
    logging.info(f"Augmentation - Scale: {args.scale}")
    logging.info(f"Augmentation - Translate: {args.translate}")
    logging.info("=" * 70)

    try:
        # Get device
        device = get_device()

        # Train model
        train_model(
            data_yaml=args.data_yaml,
            output_dir=args.output_dir,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            device=device,
            fliplr=args.fliplr,
            scale=args.scale,
            translate=args.translate,
            lr0=args.lr0,
            lrf=args.lrf,
            warmup_epochs=args.warmup_epochs,
            patience=args.patience,
        )

        # Print summary
        print_training_summary(args.output_dir)

        logging.info("=" * 70)
        logging.info("Pipeline completed successfully!")
        logging.info("=" * 70)

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        import traceback

        logging.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
