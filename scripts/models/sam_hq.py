"""Script to run SAM HQ segmentation on snow leopard images using Grounding DINO predictions.

This script performs high-quality instance segmentation by combining object detection
from Grounding DINO with precise mask generation from SAM HQ. It processes snow leopard images
to produce detailed segmentation masks for downstream analysis and re-identification.

This script:
1. Loads SAM HQ model from checkpoint (vit_b, vit_l, or vit_h)
2. Reads Grounding DINO predictions (bounding boxes) from JSON files
3. Runs SAM HQ segmentation using bounding box prompts
4. Saves binary masks as PNG files
5. Generates predictions JSON with mask metadata
6. Creates visualizations with colored overlays and scores
7. Uses GPU when available for faster inference

Usage:
    python scripts/models/sam_hq.py --input-dir <path> --predictions-dir <path> --output-dir <path> --checkpoint <path> [options]

Example:
    python scripts/models/sam_hq.py \
        --input-dir "./data/02_processed/locations" \
        --predictions-dir "./data/05_model_output/grounding_dino/predictions" \
        --output-dir "./data/05_model_output/sam_hq" \
        --checkpoint "./data/04_models/SAM_HQ/sam_hq_vit_b.pth" \
        --model-type vit_b

Technical Details:
    - Uses segment_anything_hq library (not HuggingFace transformers)
    - Employs HQ token for superior mask quality vs standard SAM
    - Processes images in location/individual/body_part directory structure
    - Outputs binary masks (0=background, 255=foreground)
    - Visualization overlays include both SAM and GDINO confidence scores

Notes:
    - Model checkpoint must match model-type (vit_b, vit_l, vit_h)
    - GPU highly recommended for processing large datasets
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything_hq import SamPredictor, sam_model_registry

from snowleopard_reid.images import resize_image_if_needed_cv2


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def get_device() -> str:
    """
    Determine the best available device (CUDA GPU or CPU).

    Returns:
        str: "cuda" or "cpu"
    """
    if torch.cuda.is_available():
        device = "cuda"
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        device = "cpu"
        logging.info("Using CPU (no GPU available)")

    return device


def load_sam_model(checkpoint_path: Path, model_type: str, device: str) -> SamPredictor:
    """
    Load the SAM HQ model.

    Args:
        checkpoint_path: Path to model checkpoint
        model_type: Model type (e.g. "vit_b", "vit_l", "vit_h")
        device: Device to load model on

    Returns:
        SamPredictor instance
    """
    logging.info(f"Loading SAM HQ model: {model_type}")

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device)

    predictor = SamPredictor(sam)

    logging.info("Model loaded successfully")
    return predictor


def find_prediction_files(predictions_dir: Path) -> list[Path]:
    """Find all prediction JSON files."""
    return sorted(predictions_dir.rglob("*.json"))


def load_grounding_dino_predictions(prediction_path: Path) -> dict[str, Any]:
    """Load Grounding DINO predictions from JSON."""
    with open(prediction_path, "r") as f:
        return json.load(f)


def run_sam_on_image(
    predictor: SamPredictor,
    image: np.ndarray,
    bbox: np.ndarray,
) -> dict[str, Any]:
    """
    Run SAM HQ inference on an image with a bounding box.

    Args:
        predictor: SAM predictor
        image: Image as numpy array (RGB)
        bbox: Bounding box in format [x1, y1, x2, y2] (pixel coordinates)

    Returns:
        Dict with keys: masks, scores, logits
    """
    # Set the image once for this image
    predictor.set_image(image)

    # Run prediction with the bounding box
    # box[None, :] adds a batch dimension
    masks, scores, logits = predictor.predict(
        box=bbox[None, :],
        multimask_output=False,
        hq_token_only=True,  # Use HQ token for best quality
    )

    return {"masks": masks, "scores": scores, "logits": logits}


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    """Save a binary mask as PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # mask is 2D boolean/binary array
    mask_2d = mask.squeeze()
    mask_uint8 = (mask_2d * 255).astype(np.uint8)

    mask_image = Image.fromarray(mask_uint8, mode="L")
    mask_image.save(output_path)


def visualize_masks(
    image: np.ndarray,
    segmentations: list,
    output_path: Path,
    alpha: float = 0.3,
) -> None:
    """
    Create visualization with all masks, bounding boxes, and labels.

    Args:
        image: Original image as numpy array (RGB)
        segmentations: List of segmentation dicts with 'mask', 'label', 'score', 'bbox_xyxyn'
        output_path: Path to save visualization
        alpha: Transparency for mask overlay (default: 0.3)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert RGB to BGR for cv2
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    orig_height, orig_width = img.shape[:2]

    # Resize image FIRST, then draw everything on the smaller image
    # This keeps text readable at the final output size
    img = resize_image_if_needed_cv2(img, max_dim=1024)
    height, width = img.shape[:2]

    # Calculate scale factor for coordinates
    scale_x = width / orig_width
    scale_y = height / orig_height

    # Colors for different instances (BGR format for cv2)
    colors = [
        (0, 255, 0),  # Green
        (0, 0, 255),  # Red
        (255, 0, 0),  # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
    ]

    # Draw each mask with different color
    for idx, seg in enumerate(segmentations):
        color = colors[idx % len(colors)]
        mask = seg["mask"]
        bbox_norm = seg["bbox_xyxyn"]
        label = seg["label"]
        sam_score = seg["score"]
        gdino_score = seg["gdino_score"]

        # Resize mask to match resized image
        mask_2d = mask.squeeze().astype(np.uint8)
        mask_resized = cv2.resize(
            mask_2d, (width, height), interpolation=cv2.INTER_NEAREST
        )
        mask_bool = mask_resized.astype(bool)

        # Create mask overlay on resized image
        overlay = img.copy()
        overlay[mask_bool] = color

        # Blend with original image
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Draw mask contour
        mask_uint8 = (mask_bool * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img, contours, -1, color, 2)

        # Draw bounding box (scaled to resized image)
        x_min = int(bbox_norm["x_min"] * orig_width * scale_x)
        y_min = int(bbox_norm["y_min"] * orig_height * scale_y)
        x_max = int(bbox_norm["x_max"] * orig_width * scale_x)
        y_max = int(bbox_norm["y_max"] * orig_height * scale_y)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 3)

        # Add label with scores (now properly sized for final image)
        label_text = f"{label} {idx + 1}: SAM={sam_score:.2f} GDINO={gdino_score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2

        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, font_thickness
        )

        # Draw text background with padding
        padding = 6
        cv2.rectangle(
            img,
            (x_min, y_min - text_height - baseline - padding * 2),
            (x_min + text_width + padding * 2, y_min),
            color,
            -1,
        )

        # Draw text
        cv2.putText(
            img,
            label_text,
            (x_min + padding, y_min - padding),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )

    # Save visualization (already resized)
    cv2.imwrite(str(output_path), img)


def process_image(
    image_path: Path,
    prediction_path: Path,
    input_dir: Path,
    output_dir: Path,
    predictor: SamPredictor,
    visualize: bool = True,
) -> None:
    """
    Process a single image with SAM HQ.

    Args:
        image_path: Path to image
        prediction_path: Path to Grounding DINO prediction JSON
        input_dir: Base input directory
        output_dir: Base output directory
        predictor: SAM predictor
        visualize: Whether to create visualizations
    """
    # Load predictions
    gdino_preds = load_grounding_dino_predictions(prediction_path)

    # Load image as numpy array (RGB)
    image = np.array(Image.open(image_path).convert("RGB"))
    height, width = image.shape[:2]

    # Get relative path for maintaining structure
    rel_path = image_path.relative_to(input_dir)

    # Create output directories
    masks_dir = output_dir / "masks" / rel_path.parent
    predictions_dir = output_dir / "predictions" / rel_path.parent
    visualizations_dir = output_dir / "visualizations" / rel_path.parent

    # Process each detection
    segmentations = []
    segmentations_with_masks = []  # For visualization

    for i, detection in enumerate(gdino_preds["detections"]):
        bbox_norm = detection["bbox_xyxyn"]

        # Convert to pixel coordinates
        x1 = bbox_norm["x_min"] * width
        y1 = bbox_norm["y_min"] * height
        x2 = bbox_norm["x_max"] * width
        y2 = bbox_norm["y_max"] * height

        bbox = np.array([x1, y1, x2, y2])

        logging.debug(f"  Processing detection {i}: bbox={bbox}")

        # Run SAM
        result = run_sam_on_image(predictor=predictor, image=image, bbox=bbox)

        mask = result["masks"][0]  # Get first mask (we use multimask_output=False)
        score = float(result["scores"][0])

        logging.debug(f"  Mask shape: {mask.shape}, Score: {score}")

        # Save mask
        mask_filename = f"{rel_path.stem}_mask_{i:03d}.png"
        mask_path = masks_dir / mask_filename
        save_mask(mask=mask, output_path=mask_path)

        seg_info = {
            "mask_file": mask_filename,
            "score": score,
            "label": detection["label"],
            "gdino_score": detection["score"],
            "bbox_xyxyn": bbox_norm,
        }
        segmentations.append(seg_info)

        # Keep mask for visualization
        segmentations_with_masks.append(
            {
                **seg_info,
                "mask": mask,
            }
        )

    # Save predictions JSON
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pred_output_path = predictions_dir / f"{rel_path.stem}.json"

    sam_predictions = {
        "image_path": str(image_path),
        "image_size": {"width": width, "height": height},
        "grounding_dino_predictions": str(prediction_path),
        "segmentations": segmentations,
    }

    with open(pred_output_path, "w") as f:
        json.dump(sam_predictions, f, indent=2)

    logging.info(f"  Generated {len(segmentations)} masks")

    # Create visualization with all masks
    if visualize and segmentations_with_masks:
        viz_output_path = visualizations_dir / f"{rel_path.stem}.jpg"
        visualize_masks(
            image=image,
            segmentations=segmentations_with_masks,
            output_path=viz_output_path,
        )
        logging.debug(f"  Saved visualization to: {viz_output_path}")


def process_all_images(
    input_dir: Path,
    predictions_dir: Path,
    output_dir: Path,
    predictor: SamPredictor,
    visualize: bool = True,
) -> None:
    """Process all images."""
    prediction_files = find_prediction_files(predictions_dir)

    if not prediction_files:
        logging.warning(f"No prediction files found in {predictions_dir}")
        return

    logging.info(f"Found {len(prediction_files)} prediction files to process")

    processed_count = 0
    for i, prediction_path in enumerate(prediction_files, 1):
        try:
            # Load predictions to get image path
            predictions = load_grounding_dino_predictions(prediction_path)
            image_path = Path(predictions["image_path"])

            if not image_path.exists():
                logging.warning(f"Image not found: {image_path}, skipping")
                continue

            logging.info(f"[{i}/{len(prediction_files)}] Processing: {image_path.name}")

            process_image(
                image_path=image_path,
                prediction_path=prediction_path,
                input_dir=input_dir,
                output_dir=output_dir,
                predictor=predictor,
                visualize=visualize,
            )

            processed_count += 1

        except Exception as e:
            logging.error(f"Failed to process {prediction_path}: {e}")
            import traceback

            logging.debug(traceback.format_exc())
            continue

    logging.info(
        f"Processing complete! Successfully processed {processed_count}/{len(prediction_files)} images"
    )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run SAM HQ segmentation using Grounding DINO predictions"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing original images",
    )

    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory containing Grounding DINO predictions (JSON files)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for SAM HQ masks and predictions",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to SAM HQ checkpoint file",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM model type (default: vit_b)",
    )

    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip creating visualizations",
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

    if not args.predictions_dir.exists():
        logging.error(f"Predictions directory not found: {args.predictions_dir}")
        sys.exit(1)

    if not args.checkpoint.exists():
        logging.error(f"Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    logging.info("=" * 70)
    logging.info("SAM HQ Segmentation Pipeline")
    logging.info("=" * 70)
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Predictions directory: {args.predictions_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Checkpoint: {args.checkpoint}")
    logging.info(f"Model type: {args.model_type}")
    logging.info("=" * 70)

    try:
        # Get device
        device = get_device()

        # Load model
        predictor = load_sam_model(
            checkpoint_path=args.checkpoint, model_type=args.model_type, device=device
        )

        # Process all images
        process_all_images(
            input_dir=args.input_dir,
            predictions_dir=args.predictions_dir,
            output_dir=args.output_dir,
            predictor=predictor,
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
