"""Convert SAM HQ segmentation outputs to YOLO segmentation format.

This script bridges the gap between SAM HQ's high-quality binary masks and YOLO's
polygon-based segmentation format, enabling training of YOLO segmentation models on
data annotated with SAM HQ. It processes entire datasets while preserving directory
structure and generating visualizations for quality control.

This script:
1. Loads SAM HQ prediction JSON files and binary mask PNGs
2. Extracts contours from binary masks using cv2.RETR_EXTERNAL
3. Simplifies polygons using cv2.CHAIN_APPROX_SIMPLE (natural simplification)
4. Filters out small or degenerate contours (configurable min area/points)
5. Normalizes polygon coordinates to [0, 1] range
6. Writes YOLO .txt annotations with format: <class_id> <x1> <y1> ... <xn> <yn>
7. Generates visualizations with polygon overlays for quality verification

Usage:
    python scripts/data/sam_to_yolo.py --sam-output-dir <path> --yolo-output-dir <path> [options]

Example:
    python scripts/data/sam_to_yolo.py \
        --sam-output-dir "./data/05_model_output/sam_hq" \
        --yolo-output-dir "./data/02_processed/yolo/segmentation" \
        --min-area 200 \
        --min-points 3 \
        --max-size 1024 \
        --skip-existing

Technical Details:
    - Uses cv2.CHAIN_APPROX_SIMPLE instead of Douglas-Peucker for natural polygon
      simplification without manual epsilon tuning
    - Selects largest contour per mask to handle multi-contour edge cases
    - Preserves location/individual/body_part directory structure from SAM HQ outputs
    - Resizes images to max_size while maintaining aspect ratio
    - All coordinates clamped to [0, 1] to prevent out-of-bounds values
    - Visualization overlays use color-coded polygons with transparency

Notes:
    - YOLO requires minimum 3 points per polygon; contours with fewer points are skipped
    - Small contours (< min_area) are filtered to remove noise and artifacts
    - Use --skip-existing for incremental processing of large datasets
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

from snowleopard_reid.images import resize_image_if_needed_cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Segmentation(BaseModel):
    """SAM HQ segmentation output for a single instance."""

    mask_file: str
    score: float
    label: str
    gdino_score: float
    bbox_xyxyn: BoundingBox


class SAMPrediction(BaseModel):
    """SAM HQ prediction JSON structure."""

    image_path: str
    image_size: dict[str, int] = Field(..., description="Width and height in pixels")
    grounding_dino_predictions: str
    segmentations: list[Segmentation]


class YOLOStats(BaseModel):
    """Statistics for YOLO dataset conversion."""

    total_images: int = 0
    total_segments: int = 0
    skipped_images: int = 0
    skipped_segments: int = 0
    avg_points_per_polygon: float = 0.0
    min_points: int = 0
    max_points: int = 0


def mask_to_polygon(
    mask_path: Path,
    image_width: int,
    image_height: int,
    min_area: int = 200,
    min_points: int = 3,
) -> list[tuple[float, float]] | None:
    """
    Convert binary mask PNG to normalized polygon coordinates.

    Uses cv2.CHAIN_APPROX_SIMPLE for natural simplification without manual epsilon tuning.

    Args:
        mask_path: Path to binary mask PNG file (0=background, 255=object)
        image_width: Original image width in pixels
        image_height: Original image height in pixels
        min_area: Minimum contour area in pixels to keep (default: 200)
        min_points: Minimum number of polygon points required (YOLO requires >= 3)

    Returns:
        List of (x, y) normalized coordinates, or None if invalid
    """
    # Load binary mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.warning(f"Failed to load mask: {mask_path}")
        return None

    # Extract contours (RETR_EXTERNAL gets only outermost contours)
    # CHAIN_APPROX_SIMPLE removes redundant points along straight lines
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.warning(f"No contours found in mask: {mask_path}")
        return None

    # Use largest contour (main object)
    largest_contour = max(contours, key=cv2.contourArea)

    # Filter by area
    area = cv2.contourArea(largest_contour)
    if area < min_area:
        logger.debug(f"Contour too small ({area:.0f} pixels): {mask_path}")
        return None

    # Convert contour to polygon format (N, 2) array
    polygon = largest_contour.reshape(-1, 2)

    # Validate minimum points
    if len(polygon) < min_points:
        logger.warning(
            f"Polygon has {len(polygon)} points (< {min_points}): {mask_path}"
        )
        return None

    # Normalize coordinates to [0, 1]
    normalized_points = []
    for point in polygon:
        x_norm = float(point[0]) / image_width
        y_norm = float(point[1]) / image_height
        # Clamp to [0, 1] range to handle any edge cases
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        normalized_points.append((x_norm, y_norm))

    return normalized_points


def create_yolo_annotation(
    sam_prediction: SAMPrediction,
    masks_dir: Path,
    rel_path: Path,
    min_area: int,
    min_points: int,
    class_id: int = 0,
) -> tuple[list[str], int]:
    """
    Create YOLO annotation lines from SAM prediction.

    Args:
        sam_prediction: Parsed SAM HQ prediction JSON
        masks_dir: Directory containing mask PNG files
        rel_path: Relative path from predictions directory (preserves directory structure)
        min_area: Minimum contour area in pixels
        min_points: Minimum polygon points required
        class_id: YOLO class ID (default: 0 for "snow leopard")

    Returns:
        Tuple of (annotation_lines, skipped_count)
    """
    annotation_lines = []
    skipped_count = 0

    img_width = sam_prediction.image_size["width"]
    img_height = sam_prediction.image_size["height"]

    for seg in sam_prediction.segmentations:
        # Locate mask file using relative path directory structure
        mask_path = masks_dir / rel_path.parent / seg.mask_file

        if not mask_path.exists():
            logger.warning(f"Mask file not found: {mask_path}")
            skipped_count += 1
            continue

        # Convert mask to polygon
        polygon = mask_to_polygon(
            mask_path=mask_path,
            image_width=img_width,
            image_height=img_height,
            min_area=min_area,
            min_points=min_points,
        )

        if polygon is None:
            skipped_count += 1
            continue

        # Format YOLO annotation: <class_id> <x1> <y1> <x2> <y2> ...
        coords_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
        annotation_lines.append(f"{class_id} {coords_str}")

    return annotation_lines, skipped_count


def create_visualization(
    image_path: Path,
    annotation_lines: list[str],
    image_size: tuple[int, int],
    output_path: Path,
    max_dim: int = 1024,
) -> None:
    """
    Create visualization with polygons and bounding boxes drawn on image.

    Args:
        image_path: Path to source image
        annotation_lines: YOLO annotation lines with polygon coordinates
        image_size: Original image dimensions (width, height) before resize
        output_path: Where to save visualization
        max_dim: Maximum dimension for resized image
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Failed to load image for visualization: {image_path}")
        return

    # Resize if needed
    img = resize_image_if_needed_cv2(img=img, max_dim=max_dim)

    # Get actual image dimensions after resize
    actual_height, actual_width = img.shape[:2]

    # Original dimensions for coordinate conversion
    img_width, img_height = image_size

    # Colors for different instances
    colors = [
        (0, 255, 0),  # Green
        (0, 0, 255),  # Red
        (255, 0, 0),  # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
    ]

    for idx, line in enumerate(annotation_lines):
        parts = line.split()

        # Parse normalized polygon coordinates
        coords = [float(x) for x in parts[1:]]
        num_points = len(coords) // 2

        # Convert to pixel coordinates (using actual resized dimensions)
        polygon_points = []
        for i in range(num_points):
            x_norm = coords[i * 2]
            y_norm = coords[i * 2 + 1]
            x_px = int(x_norm * actual_width)
            y_px = int(y_norm * actual_height)
            polygon_points.append((x_px, y_px))

        # Draw polygon
        polygon_np = np.array(polygon_points, dtype=np.int32)
        color = colors[idx % len(colors)]

        # Draw filled polygon with transparency
        overlay = img.copy()
        cv2.fillPoly(overlay, [polygon_np], color)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        # Draw polygon outline
        cv2.polylines(img, [polygon_np], isClosed=True, color=color, thickness=2)

        # Draw bounding box
        x_coords = [p[0] for p in polygon_points]
        y_coords = [p[1] for p in polygon_points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        # Add label
        label = f"leopard {idx + 1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        # Draw text background
        cv2.rectangle(
            img,
            (x_min, y_min - text_height - 10),
            (x_min + text_width + 10, y_min),
            color,
            -1,
        )
        cv2.putText(
            img,
            label,
            (x_min + 5, y_min - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )

    # Save visualization
    cv2.imwrite(str(output_path), img)


def convert_sam_to_yolo(
    sam_output_dir: Path,
    yolo_output_dir: Path,
    min_area: int,
    min_points: int,
    max_size: int,
    skip_existing: bool,
) -> tuple[YOLOStats, Path]:
    """
    Convert SAM HQ outputs to YOLO segmentation format.

    Args:
        sam_output_dir: SAM HQ output directory
        yolo_output_dir: YOLO dataset base directory
        min_area: Minimum contour area in pixels
        min_points: Minimum polygon points
        max_size: Maximum image dimension (images will be resized if larger)
        skip_existing: Skip already processed images

    Returns:
        Tuple of (YOLOStats with conversion statistics, Path to dataset output directory)
    """
    # Setup directories
    predictions_dir = sam_output_dir / "predictions"
    masks_dir = sam_output_dir / "masks"

    images_dir = yolo_output_dir / "images"
    labels_dir = yolo_output_dir / "labels"
    visualizations_dir = yolo_output_dir / "visualizations"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    # Find all prediction JSON files
    prediction_files = sorted(predictions_dir.rglob("*.json"))
    logger.info(f"Found {len(prediction_files)} prediction files")

    if not prediction_files:
        logger.warning(f"No prediction files found in {predictions_dir}")
        return YOLOStats(), yolo_output_dir

    # Process each prediction
    stats = YOLOStats()
    all_point_counts = []

    for pred_file in prediction_files:
        # Load prediction JSON
        try:
            with open(pred_file, "r") as f:
                pred_data = json.load(f)
            sam_pred = SAMPrediction(**pred_data)
        except Exception as e:
            logger.error(f"Failed to parse {pred_file}: {e}")
            stats.skipped_images += 1
            continue

        # Get relative path from predictions directory (preserves location/individual/body_part)
        rel_path = pred_file.relative_to(predictions_dir)

        # Get original image path
        original_image_path = Path(sam_pred.image_path)

        # Preserve full directory structure
        output_image_subdir = images_dir / rel_path.parent
        output_label_subdir = labels_dir / rel_path.parent
        output_viz_subdir = visualizations_dir / rel_path.parent

        output_image_subdir.mkdir(parents=True, exist_ok=True)
        output_label_subdir.mkdir(parents=True, exist_ok=True)
        output_viz_subdir.mkdir(parents=True, exist_ok=True)

        # Use original filename
        image_filename = original_image_path.name

        # Output paths
        output_image = output_image_subdir / image_filename
        output_label = output_label_subdir / f"{original_image_path.stem}.txt"
        output_viz = output_viz_subdir / image_filename

        # Skip if exists
        if skip_existing and output_image.exists() and output_label.exists():
            logger.debug(f"Skipping existing: {rel_path}")
            continue

        # Check if source image exists
        if not original_image_path.exists():
            logger.warning(f"Source image not found: {original_image_path}")
            stats.skipped_images += 1
            continue

        # Create YOLO annotation
        annotation_lines, skipped_segs = create_yolo_annotation(
            sam_prediction=sam_pred,
            masks_dir=masks_dir,
            rel_path=rel_path,
            min_area=min_area,
            min_points=min_points,
            class_id=0,  # Single "snow leopard" class
        )

        stats.skipped_segments += skipped_segs

        if not annotation_lines:
            logger.warning(f"No valid annotations for {rel_path}")
            stats.skipped_images += 1
            continue

        # Load, resize, and save image
        try:
            img = cv2.imread(str(original_image_path))
            if img is None:
                logger.error(f"Failed to load image: {original_image_path}")
                stats.skipped_images += 1
                continue

            # Resize if needed
            img = resize_image_if_needed_cv2(img, max_dim=max_size)

            # Save resized image
            cv2.imwrite(str(output_image), img)
        except Exception as e:
            logger.error(f"Failed to process {original_image_path}: {e}")
            stats.skipped_images += 1
            continue

        # Write annotation
        try:
            with open(output_label, "w") as f:
                f.write("\n".join(annotation_lines) + "\n")
        except Exception as e:
            logger.error(f"Failed to write {output_label}: {e}")
            stats.skipped_images += 1
            continue

        # Create visualization
        try:
            img_width = sam_pred.image_size["width"]
            img_height = sam_pred.image_size["height"]
            create_visualization(
                image_path=output_image,
                annotation_lines=annotation_lines,
                image_size=(img_width, img_height),
                output_path=output_viz,
                max_dim=max_size,
            )
        except Exception as e:
            logger.warning(f"Failed to create visualization for {image_filename}: {e}")

        # Update statistics
        stats.total_images += 1
        stats.total_segments += len(annotation_lines)

        # Track point counts for statistics
        for line in annotation_lines:
            parts = line.split()
            # First part is class_id, rest are x,y pairs
            num_points = (len(parts) - 1) // 2
            all_point_counts.append(num_points)

        if stats.total_images % 50 == 0:
            logger.info(f"Processed {stats.total_images} images...")

    # Calculate point statistics
    if all_point_counts:
        stats.avg_points_per_polygon = np.mean(all_point_counts)
        stats.min_points = min(all_point_counts)
        stats.max_points = max(all_point_counts)

    return stats, yolo_output_dir


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert SAM HQ segmentation outputs to YOLO format"
    )
    parser.add_argument(
        "--sam-output-dir",
        type=Path,
        required=True,
        help="SAM HQ output directory (e.g., data/05_model_output/sam_hq)",
    )
    parser.add_argument(
        "--yolo-output-dir",
        type=Path,
        required=True,
        help="YOLO dataset output directory (e.g., data/02_processed/yolo/segmentation)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=200,
        help="Minimum contour area in pixels (default: 200)",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="Minimum polygon points (YOLO requires >= 3)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help="Maximum image dimension in pixels (default: 1024)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already exist in output",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate inputs
    if not args.sam_output_dir.exists():
        logger.error(f"SAM output directory not found: {args.sam_output_dir}")
        return

    predictions_dir = args.sam_output_dir / "predictions"
    masks_dir = args.sam_output_dir / "masks"

    if not predictions_dir.exists():
        logger.error(f"Predictions directory not found: {predictions_dir}")
        return

    if not masks_dir.exists():
        logger.error(f"Masks directory not found: {masks_dir}")
        return

    logger.info("Starting SAM HQ to YOLO conversion...")
    logger.info(f"SAM output dir: {args.sam_output_dir}")
    logger.info(f"YOLO output dir: {args.yolo_output_dir}")
    logger.info(f"Min area: {args.min_area} pixels")
    logger.info(f"Min points: {args.min_points}")
    logger.info(f"Max image size: {args.max_size} pixels")

    # Convert dataset
    stats, dataset_output_dir = convert_sam_to_yolo(
        sam_output_dir=args.sam_output_dir,
        yolo_output_dir=args.yolo_output_dir,
        min_area=args.min_area,
        min_points=args.min_points,
        max_size=args.max_size,
        skip_existing=args.skip_existing,
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("Conversion Complete!")
    logger.info(f"Total images: {stats.total_images}")
    logger.info(f"Total segments: {stats.total_segments}")
    logger.info(f"Skipped images: {stats.skipped_images}")
    logger.info(f"Skipped segments: {stats.skipped_segments}")

    if stats.total_segments > 0:
        logger.info(f"Avg points per polygon: {stats.avg_points_per_polygon:.1f}")
        logger.info(f"Min/Max points: {stats.min_points}/{stats.max_points}")
        logger.info(
            f"Segments per image: {stats.total_segments / stats.total_images:.2f}"
        )

    logger.info(f"Output directory: {dataset_output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
