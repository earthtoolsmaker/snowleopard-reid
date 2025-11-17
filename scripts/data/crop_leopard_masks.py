"""Crop snow leopard images using SAM HQ segmentation masks.

This script processes SAM HQ segmentation outputs to extract individual snow leopard images
by cropping to mask bounding boxes with configurable padding. Each cropped image
isolates a single leopard with background masked to black, ideal for catalog building
or feature extraction where background should not influence matching.

This script:
1. Loads SAM HQ prediction JSON files with mask references
2. Reads corresponding binary mask PNG files
3. Computes tight bounding box around each mask
4. Applies configurable padding to bounding box
5. Crops original image to padded bbox and applies mask (black background)
6. Saves cropped images preserving location/individual/body_part directory structure

Usage:
    python scripts/data/crop_leopard_masks.py --sam-output-dir <path> --output-dir <path> [options]

Example:
    python scripts/data/crop_leopard_masks.py \
        --sam-output-dir "./data/05_model_output/sam_hq" \
        --output-dir "./data/02_processed/cropped_leopards" \
        --padding 10

Notes:
    - Padding prevents loss of leopard edges and provides context for feature extraction
    - Black background (0,0,0) ensures only leopard pixels contribute to features
    - Maintains location/individual/body_part hierarchy for catalog organization
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from snowleopard_reid.masks import (
    add_padding_to_bbox,
    crop_and_mask_image,
    get_mask_bbox,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def process_prediction_file(
    prediction_path: Path,
    sam_output_dir: Path,
    output_dir: Path,
    padding: int,
    verbose: bool = False,
) -> int:
    """
    Process a single SAM HQ prediction file and generate cropped images.

    Args:
        prediction_path: Path to prediction JSON file
        sam_output_dir: Root directory of SAM HQ outputs
        output_dir: Output directory for cropped images
        padding: Padding in pixels around mask bbox
        verbose: Enable verbose logging

    Returns:
        Number of cropped images generated
    """
    if verbose:
        logger.info(f"Processing {prediction_path}")

    # Load prediction JSON
    with open(prediction_path) as f:
        prediction = json.load(f)

    # Load original image
    original_image_path = Path(prediction["image_path"])
    if not original_image_path.exists():
        logger.warning(f"Original image not found: {original_image_path}")
        return 0

    image = Image.open(original_image_path)
    image_width, image_height = image.size

    # Get relative path structure (location/individual/body_part)
    # prediction_path is: predictions/<location>/<individual>/<body_part>/<image>.json
    rel_path = prediction_path.relative_to(sam_output_dir / "predictions")

    # Preserve the full directory structure
    output_subdir = output_dir / rel_path.parent
    output_subdir.mkdir(parents=True, exist_ok=True)

    image_stem = prediction_path.stem

    # Process each segmentation
    count = 0
    for idx, segmentation in enumerate(prediction["segmentations"]):
        mask_filename = segmentation["mask_file"]
        mask_path = sam_output_dir / "masks" / rel_path.parent / mask_filename

        if not mask_path.exists():
            logger.warning(f"Mask not found: {mask_path}")
            continue

        # Load mask
        mask_image = Image.open(mask_path)
        mask_array = np.array(mask_image)

        try:
            # Calculate tight bounding box from mask
            bbox = get_mask_bbox(mask_array)

            # Add padding
            padded_bbox = add_padding_to_bbox(
                bbox=bbox,
                padding=padding,
                image_width=image_width,
                image_height=image_height,
            )

            # Crop and apply mask
            cropped = crop_and_mask_image(
                image=image, mask=mask_array, bbox=padded_bbox
            )

            # Save cropped image
            output_filename = f"{image_stem}_{idx:03d}.jpg"
            output_path = output_subdir / output_filename
            cropped.save(output_path, quality=95)

            if verbose:
                logger.info(f"  Saved {output_path}")

            count += 1

        except ValueError as e:
            logger.warning(f"Skipping {mask_filename}: {e}")
            continue

    return count


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Crop snow leopard images using SAM HQ segmentation masks"
    )
    parser.add_argument(
        "--sam-output-dir",
        type=Path,
        required=True,
        help="SAM HQ output directory (contains predictions/ and masks/ subdirs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for cropped images",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=5,
        help="Padding in pixels around mask bounding box (default: 5)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Validate input directory
    predictions_dir = args.sam_output_dir / "predictions"
    masks_dir = args.sam_output_dir / "masks"

    if not predictions_dir.exists():
        logger.error(f"Predictions directory not found: {predictions_dir}")
        return 1

    if not masks_dir.exists():
        logger.error(f"Masks directory not found: {masks_dir}")
        return 1

    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f"SAM output directory: {args.sam_output_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Padding: {args.padding}px")

    # Find all prediction JSON files
    prediction_files = list(predictions_dir.rglob("*.json"))
    logger.info(f"Found {len(prediction_files)} prediction files")

    if not prediction_files:
        logger.warning("No prediction files found")
        return 0

    # Process all prediction files
    total_cropped = 0
    total_files = len(prediction_files)
    for idx, prediction_path in enumerate(sorted(prediction_files), start=1):
        logger.info(f"Processing file {idx}/{total_files}: {prediction_path.name}")
        count = process_prediction_file(
            prediction_path=prediction_path,
            sam_output_dir=args.sam_output_dir,
            output_dir=args.output_dir,
            padding=args.padding,
            verbose=args.verbose,
        )
        total_cropped += count
        logger.info(f"  Generated {count} cropped images from this file")

    logger.info(f"Successfully created {total_cropped} cropped images")
    return 0


if __name__ == "__main__":
    exit(main())
