"""Script to run Grounding DINO object detection on snow leopard images.

This script performs zero-shot object detection using Grounding DINO, a vision-language
model that detects objects based on text prompts. It processes snow leopard images to generate
bounding box predictions for downstream analysis and identification.

This script:
1. Loads Grounding DINO model from HuggingFace
2. Processes all images in input directory (preserving location/individual/body_part structure)
3. Runs object detection with configurable text prompts
4. Saves predictions as JSON files with normalized bounding boxes
5. Creates visualizations with labeled bounding boxes
6. Uses GPU when available for faster inference

Usage:
    python scripts/models/grounding_dino.py --input-dir <path> --output-dir <path> [options]

Example:
    python scripts/models/grounding_dino.py \
        --input-dir "./data/02_processed/locations/naryn" \
        --output-dir "./data/05_model_output/grounding_dino/naryn" \
        --text-prompt "a snow leopard." \
        --model-id "IDEA-Research/grounding-dino-base" \
        --box-threshold 0.30 \
        --text-threshold 0.20

Notes:
    - Text prompts should end with period and use specific object descriptions
    - Model auto-downloads from HuggingFace on first run
    - Lower thresholds recommended for snow leopards due to camouflage
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from snowleopard_reid.images import resize_image_if_needed


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def get_device() -> torch.device:
    """
    Determine the best available device (CUDA GPU or CPU).

    Returns:
        torch.device: The device to use for inference
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        device = torch.device("cpu")
        logging.info("Using CPU (no GPU available)")

    return device


def load_model(model_id: str, device: torch.device) -> tuple[Any, Any]:
    """
    Load the Grounding DINO model and processor.

    Args:
        model_id: HuggingFace model identifier
        device: Device to load the model on

    Returns:
        Tuple of (processor, model)
    """
    logging.info(f"Loading model: {model_id}")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model = model.to(device)
    model.eval()

    logging.info("Model loaded successfully")
    return processor, model


def find_images(input_dir: Path) -> list[Path]:
    """
    Find all image files in the input directory recursively.

    Args:
        input_dir: Directory to search

    Returns:
        List of image file paths
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    images = []

    for ext in image_extensions:
        images.extend(input_dir.rglob(f"*{ext}"))
        images.extend(input_dir.rglob(f"*{ext.upper()}"))

    return sorted(set(images))


def run_inference(
    image_path: Path,
    processor: Any,
    model: Any,
    device: torch.device,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> dict[str, Any]:
    """
    Run Grounding DINO inference on a single image.

    Args:
        image_path: Path to the image
        processor: The processor for input preparation
        model: The Grounding DINO model
        device: Device to run inference on
        text_prompt: Text prompt describing objects to detect
        box_threshold: Confidence threshold for bounding boxes
        text_threshold: Confidence threshold for text matching

    Returns:
        Dictionary containing predictions with boxes, scores, and labels
    """
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Prepare inputs
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process outputs
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],  # (height, width)
    )[0]

    # Convert to serializable format
    width, height = image.size
    predictions = {
        "image_path": str(image_path),
        "image_size": {"width": width, "height": height},
        "text_prompt": text_prompt,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "detections": [],
    }

    # Add detections (use text_labels for string labels)
    labels = results.get("text_labels", results.get("labels", []))
    for box, score, label in zip(results["boxes"], results["scores"], labels):
        predictions["detections"].append(
            {
                "bbox_xyxyn": {
                    "x_min": float(box[0]) / width,
                    "y_min": float(box[1]) / height,
                    "x_max": float(box[2]) / width,
                    "y_max": float(box[3]) / height,
                },
                "score": float(score),
                "label": label,
            }
        )

    return predictions


def visualize_predictions(
    image_path: Path,
    predictions: dict[str, Any],
    output_path: Path,
    show_labels: bool = True,
    line_width: int = 3,
) -> None:
    """
    Create a visualization of predictions with bounding boxes.

    Args:
        image_path: Path to the original image
        predictions: Predictions dictionary from run_inference
        output_path: Path to save the visualization
        show_labels: Whether to show labels on the visualization
        line_width: Width of bounding box lines
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
    except Exception:
        font = ImageFont.load_default()

    # Define colors for different labels
    colors = [
        "#FF0000",
        "#00FF00",
        "#0000FF",
        "#FFFF00",
        "#FF00FF",
        "#00FFFF",
        "#FFA500",
        "#800080",
        "#FFC0CB",
        "#A52A2A",
    ]

    # Draw detections
    for i, detection in enumerate(predictions["detections"]):
        bbox_norm = detection["bbox_xyxyn"]
        score = detection["score"]
        label = detection["label"]

        # Convert normalized coordinates to pixel coordinates
        x_min = bbox_norm["x_min"] * width
        y_min = bbox_norm["y_min"] * height
        x_max = bbox_norm["x_max"] * width
        y_max = bbox_norm["y_max"] * height

        # Get color for this label
        color = colors[i % len(colors)]

        # Draw bounding box
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)], outline=color, width=line_width
        )

        # Draw label and score
        if show_labels:
            text = f"{label}: {score:.2f}"

            # Get text size for background rectangle
            bbox = draw.textbbox((x_min, y_min), text, font=font)

            # Draw background rectangle for text
            draw.rectangle(
                [(bbox[0] - 2, bbox[1] - 2), (bbox[2] + 2, bbox[3] + 2)], fill=color
            )

            # Draw text
            draw.text((x_min, y_min), text, fill="white", font=font)

    # Resize image if needed to reduce disk usage
    image = resize_image_if_needed(image, max_dim=1024)

    # Save visualization
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def process_images(
    input_dir: Path,
    output_dir: Path,
    processor: Any,
    model: Any,
    device: torch.device,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    visualize: bool = True,
) -> None:
    """
    Process all images in the input directory.

    Args:
        input_dir: Input directory with location/individual/body_part/image.jpg structure
        output_dir: Output directory for predictions and visualizations
        processor: The processor for input preparation
        model: The Grounding DINO model
        device: Device to run inference on
        text_prompt: Text prompt describing objects to detect
        box_threshold: Confidence threshold for bounding boxes
        text_threshold: Confidence threshold for text matching
        visualize: Whether to create visualizations
    """
    # Find all images
    images = find_images(input_dir)

    if not images:
        logging.warning(f"No images found in {input_dir}")
        return

    logging.info(f"Found {len(images)} images to process")

    # Create output directories
    predictions_dir = output_dir / "predictions"
    visualizations_dir = output_dir / "visualizations"

    predictions_dir.mkdir(parents=True, exist_ok=True)
    if visualize:
        visualizations_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    for i, image_path in enumerate(images, 1):
        try:
            # Get relative path to maintain location/individual/body_part structure
            rel_path = image_path.relative_to(input_dir)

            # Create output paths maintaining the same structure
            pred_output_path = (
                predictions_dir / rel_path.parent / f"{rel_path.stem}.json"
            )
            viz_output_path = (
                visualizations_dir / rel_path.parent / f"{rel_path.stem}.jpg"
            )

            logging.info(f"[{i}/{len(images)}] Processing: {rel_path}")

            # Run inference
            predictions = run_inference(
                image_path=image_path,
                processor=processor,
                model=model,
                device=device,
                text_prompt=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            # Save predictions
            pred_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pred_output_path, "w") as f:
                json.dump(predictions, f, indent=2)

            logging.debug(f"  Saved predictions to: {pred_output_path}")
            logging.info(f"  Detected {len(predictions['detections'])} objects")

            # Create visualization
            if visualize:
                visualize_predictions(
                    image_path=image_path,
                    predictions=predictions,
                    output_path=viz_output_path,
                )
                logging.debug(f"  Saved visualization to: {viz_output_path}")

        except Exception as e:
            logging.error(f"Failed to process {image_path}: {e}")
            continue

    logging.info(f"Processing complete! Processed {len(images)} images")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run Grounding DINO object detection on snow leopard images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s --input-dir "./data/02_processed/locations/naryn" \\
           --output-dir "./data/05_model_output/grounding_dino/naryn" \\
           --text-prompt "a snow leopard."
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing images (with location/individual/body_part structure)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for predictions and visualizations",
    )

    parser.add_argument(
        "--text-prompt",
        type=str,
        default="a snow leopard.",
        help="Text prompt describing objects to detect (separate classes with periods)",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="HuggingFace model identifier (default: IDEA-Research/grounding-dino-base)",
    )

    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.30,
        help="Confidence threshold for bounding boxes (default: 0.30)",
    )

    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.20,
        help="Confidence threshold for text matching (default: 0.20)",
    )

    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip creating visualizations (only save predictions)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Validate inputs
    if not args.input_dir.exists():
        logging.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    if not args.input_dir.is_dir():
        logging.error(f"Input path is not a directory: {args.input_dir}")
        sys.exit(1)

    if not (0.0 <= args.box_threshold <= 1.0):
        logging.error("Box threshold must be between 0.0 and 1.0")
        sys.exit(1)

    if not (0.0 <= args.text_threshold <= 1.0):
        logging.error("Text threshold must be between 0.0 and 1.0")
        sys.exit(1)

    logging.info("=" * 70)
    logging.info("Grounding DINO Object Detection Pipeline")
    logging.info("=" * 70)
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Text prompt: {args.text_prompt}")
    logging.info(f"Model: {args.model_id}")
    logging.info(f"Box threshold: {args.box_threshold}")
    logging.info(f"Text threshold: {args.text_threshold}")
    logging.info("=" * 70)

    try:
        # Get device
        device = get_device()

        # Load model
        processor, model = load_model(model_id=args.model_id, device=device)

        # Process images
        process_images(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            processor=processor,
            model=model,
            device=device,
            text_prompt=args.text_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            visualize=not args.no_visualize,
        )

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
