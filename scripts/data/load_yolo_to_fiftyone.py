"""Load YOLO segmentation dataset into FiftyOne for visual curation and quality control.

This script imports YOLO datasets into FiftyOne's interactive web interface, enabling
visual review, filtering, and tagging of images before training. It enriches samples
with metadata from SAM HQ and Grounding DINO to facilitate quality-based filtering.

This script:
1. Scans YOLO images and labels directories for location/individual/body_part structure
2. Parses YOLO polygon annotations into FiftyOne polyline format
3. Extracts SAM HQ and Grounding DINO confidence scores from prediction JSONs
4. Creates persistent FiftyOne dataset with rich metadata fields
5. Optionally launches FiftyOne web UI for immediate curation

Usage:
    python scripts/data/load_yolo_to_fiftyone.py [options]

Example:
    python scripts/data/load_yolo_to_fiftyone.py \
        --dataset-name snowleopard_yolo_segmentation \
        --images-dir ./data/02_processed/yolo/segmentation/images \
        --labels-dir ./data/02_processed/yolo/segmentation/labels \
        --sam-predictions ./data/05_model_output/sam_hq/predictions \
        --overwrite \
        --launch

Workflow:
    1. Run this script to load dataset into FiftyOne
    2. Launch FiftyOne UI: `fiftyone app launch` or use --launch flag
    3. Review images with segmentation overlays
    4. Filter by location, individual, body_part, or confidence scores (sam_score, gdino_score)
    5. Tag images to keep by adding 'selected' tag in UI
    6. Export curated selection: `make fiftyone-export`

Notes:
    - Dataset is persistent and only needs to be loaded once
    - Use --overwrite to rebuild dataset from scratch
    - Missing SAM predictions result in null scores (warnings shown)
    - All samples start with no tags; tagging is done interactively in UI
"""

import argparse
import json
from pathlib import Path

import fiftyone as fo


def parse_yolo_segmentation(label_path: Path) -> list[list[tuple[float, float]]]:
    """
    Parse YOLO segmentation format file.

    Returns list of polygons, where each polygon is a list of (x, y) normalized coordinates.
    """
    polygons = []

    if not label_path.exists():
        return polygons

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class_id + at least 3 points (6 coords)
                continue

            # Skip class_id (first element)
            coords = [float(x) for x in parts[1:]]

            # Group into (x, y) pairs
            points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            polygons.append(points)

    return polygons


def get_sam_metadata(
    image_path: Path, sam_predictions_dir: Path, images_root: Path
) -> dict:
    """
    Extract SAM HQ and Grounding DINO scores from prediction JSON.

    Returns dict with sam_score, gdino_score, or empty dict if not found.
    """
    # Construct path to SAM HQ prediction JSON
    # image_path: data/02_processed/yolo/segmentation/images/naryn/kelemis/right_flank/IMG_1234.jpg
    # images_root: data/02_processed/yolo/segmentation/images
    # relative_path: naryn/kelemis/right_flank/IMG_1234.jpg
    # sam_path: data/05_model_output/sam_hq/predictions/naryn/kelemis/right_flank/IMG_1234.json

    relative_path = image_path.relative_to(images_root)
    relative_parts = relative_path.parts
    location = relative_parts[0]
    individual = relative_parts[1]
    body_part = relative_parts[2]
    image_stem = image_path.stem

    sam_json_path = (
        sam_predictions_dir / location / individual / body_part / f"{image_stem}.json"
    )

    if not sam_json_path.exists():
        return {}

    try:
        with open(sam_json_path) as f:
            data = json.load(f)

        # Get first segmentation scores (we expect single snow leopard per image mostly)
        if data.get("segmentations"):
            seg = data["segmentations"][0]
            return {
                "sam_score": seg.get("score", 0.0),
                "gdino_score": seg.get("gdino_score", 0.0),
            }
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

    return {}


def extract_metadata(image_path: Path, images_root: Path) -> dict[str, str]:
    """
    Extract location, individual, and body_part from hierarchical path.

    Path structure: images/<location>/<individual>/<body_part>/<image>.jpg
    """
    relative_parts = image_path.relative_to(images_root).parts
    location = relative_parts[0]
    individual = relative_parts[1]
    body_part = relative_parts[2]
    return {
        "location": location,
        "individual": individual,
        "body_part": body_part,
    }


def create_fiftyone_dataset(
    dataset_name: str,
    images_dir: Path,
    labels_dir: Path,
    sam_predictions_dir: Path,
    overwrite: bool = False,
) -> fo.Dataset:
    """
    Create FiftyOne dataset from YOLO segmentation data with metadata.
    """
    # Delete existing dataset if overwrite is True
    if overwrite and fo.dataset_exists(dataset_name):
        print(f"Deleting existing dataset '{dataset_name}'...")
        fo.delete_dataset(dataset_name)

    # Create or load dataset
    if fo.dataset_exists(dataset_name):
        print(f"Loading existing dataset '{dataset_name}'...")
        dataset = fo.load_dataset(dataset_name)
        print(f"Existing dataset has {len(dataset)} samples")
        return dataset

    print(f"Creating new dataset '{dataset_name}'...")
    dataset = fo.Dataset(name=dataset_name, persistent=True)

    # Find all images
    image_paths = sorted(images_dir.rglob("*.jpg"))
    print(f"Found {len(image_paths)} images")

    samples = []
    images_without_labels = 0

    for image_path in image_paths:
        # Extract metadata from path
        metadata = extract_metadata(image_path=image_path, images_root=images_dir)

        # Construct corresponding label path
        relative_path = image_path.relative_to(images_dir)
        label_path = labels_dir / relative_path.with_suffix(".txt")

        # Parse YOLO annotations
        polygons = parse_yolo_segmentation(label_path)
        has_annotation = len(polygons) > 0

        if not has_annotation:
            images_without_labels += 1

        # Get SAM metadata
        sam_metadata = get_sam_metadata(
            image_path=image_path,
            sam_predictions_dir=sam_predictions_dir,
            images_root=images_dir,
        )

        # Create FiftyOne sample
        sample = fo.Sample(filepath=str(image_path))

        # Add custom metadata fields
        sample["location"] = metadata["location"]
        sample["individual"] = metadata["individual"]
        sample["body_part"] = metadata["body_part"]
        sample["has_annotation"] = has_annotation
        sample["sam_score"] = sam_metadata.get("sam_score")
        sample["gdino_score"] = sam_metadata.get("gdino_score")

        # Add segmentation annotations
        if has_annotation:
            detections = []
            for polygon in polygons:
                # Convert to FiftyOne polyline format
                polyline = fo.Polyline(
                    points=[polygon],  # List of polygons (we have one per detection)
                    closed=True,
                    filled=True,
                    label="snow_leopard",
                )
                detections.append(polyline)

            sample["segmentation"] = fo.Polylines(polylines=detections)

            # Store polygon complexity (average points per polygon)
            total_points = sum(len(p) for p in polygons)
            sample["polygon_points"] = total_points // len(polygons) if polygons else 0
        else:
            sample["polygon_points"] = 0

        samples.append(sample)

    # Add all samples to dataset
    print(f"Adding {len(samples)} samples to dataset...")
    dataset.add_samples(samples)

    # Print summary
    print("\nDataset created successfully!")
    print(f"  Total images: {len(dataset)}")
    print(f"  Images with annotations: {len(dataset) - images_without_labels}")
    print(f"  Images without annotations: {images_without_labels}")
    print(f"  Unique locations: {len(dataset.distinct('location'))}")
    print(f"  Unique individuals: {len(dataset.distinct('individual'))}")
    print(f"  Body parts: {sorted(dataset.distinct('body_part'))}")

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Load YOLO segmentation dataset into FiftyOne"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="snowleopard_yolo_segmentation",
        help="Name for the FiftyOne dataset (default: snowleopard_yolo_segmentation)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/02_processed/yolo/segmentation/images"),
        help="Path to images directory",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data/02_processed/yolo/segmentation/labels"),
        help="Path to YOLO labels directory",
    )
    parser.add_argument(
        "--sam-predictions",
        type=Path,
        default=Path("data/05_model_output/sam_hq/predictions"),
        help="Path to SAM HQ predictions directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset if it exists",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch FiftyOne app after loading dataset",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.images_dir.exists():
        print(f"Error: Images directory not found: {args.images_dir}")
        return

    if not args.labels_dir.exists():
        print(f"Error: Labels directory not found: {args.labels_dir}")
        return

    if not args.sam_predictions.exists():
        print(f"Warning: SAM predictions directory not found: {args.sam_predictions}")
        print("Continuing without SAM metadata...")

    # Create dataset
    dataset = create_fiftyone_dataset(
        dataset_name=args.dataset_name,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        sam_predictions_dir=args.sam_predictions,
        overwrite=args.overwrite,
    )

    print(f"\nDataset '{args.dataset_name}' is ready!")
    print("\n" + "=" * 70)
    print("CURATION WORKFLOW")
    print("=" * 70)
    print("\n1. All images start with NO tags")
    print("\n2. In the FiftyOne UI:")
    print("   - Review images and their segmentation overlays")
    print("   - Filter by location, individual, body_part, or quality scores")
    print("   - For images you want to KEEP: select them and add the 'selected' tag")
    print("   - Use the Tags panel on the right or the Tags button to add tags")
    print("\n3. After curation, export selected images:")
    print("   - Run: make fiftyone-export")
    print("   - Only images with 'selected' tag will be exported")
    print("\n4. Launch FiftyOne UI:")
    print("   - Run: fiftyone app launch")
    print("   - Or use: make fiftyone-load (to reload and launch)")
    print("=" * 70)

    if args.launch:
        print("\nLaunching FiftyOne app...")
        session = fo.launch_app(dataset)
        session.wait()


if __name__ == "__main__":
    main()
