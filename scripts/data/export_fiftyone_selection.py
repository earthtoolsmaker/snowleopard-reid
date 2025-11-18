"""Export tagged samples from FiftyOne dataset to JSON for training.

This script exports curated image selections from FiftyOne's interactive UI to a
structured JSON file suitable for dataset splitting and training. It bridges the
gap between visual curation (tagging in FiftyOne) and automated processing
(YOLO dataset preparation).

Workflow:
    1. Load dataset into FiftyOne (all samples start with no tags)
    2. Review images in FiftyOne UI
    3. Add 'selected' tag to images you want to keep
    4. Run this script to export tagged images to JSON

Usage:
    python scripts/data/export_fiftyone_selection.py [options]

Examples:
    # Export samples with 'selected' tag (default workflow):
    python scripts/data/export_fiftyone_selection.py

    # Or use make command:
    make fiftyone-export

    # Specify custom output directory:
    python scripts/data/export_fiftyone_selection.py \
        --dir-save data/my_custom_dir

    # Export all samples (no tag filtering):
    python scripts/data/export_fiftyone_selection.py --selection-mode all

Notes:
    - Only images with 'selected' tag are exported by default
    - Output JSON includes metadata: location, individual, body_part, SAM/GDINO scores
    - JSON format matches input expected by split_yolo_dataset.py
    - Selection percentages and individual breakdowns printed for verification
"""

import argparse
import json
from pathlib import Path

import fiftyone as fo


def get_project_root() -> Path:
    """Get the project root directory."""
    # Assume script is in scripts/data/ so project root is two levels up
    return Path(__file__).resolve().parent.parent.parent


def to_relative_path(absolute_path: Path, project_root: Path) -> str:
    """Convert absolute path to relative path from project root."""
    try:
        rel_path = absolute_path.relative_to(project_root)
        return f"./{rel_path}"
    except ValueError:
        # If path is not relative to project root, return as-is
        return str(absolute_path)


def export_selection(
    dataset_name: str,
    output_json: Path,
    selection_mode: str = "selected",
) -> None:
    """
    Export selected samples from FiftyOne dataset.

    Args:
        dataset_name: Name of the FiftyOne dataset
        output_json: Path to save JSON file
        selection_mode: How to determine which samples to export
            - "selected": Export only selected samples (samples.selected())
            - "all": Export all samples
            - "with_annotations": Export only samples with annotations
    """
    # Load dataset
    if not fo.dataset_exists(dataset_name):
        print(f"Error: Dataset '{dataset_name}' not found")
        print("Available datasets:", fo.list_datasets())
        return

    print(f"Loading dataset '{dataset_name}'...")
    dataset = fo.load_dataset(dataset_name)

    # Filter samples based on selection mode
    if selection_mode == "selected":
        # Get selected samples using tags
        # In FiftyOne, persistent selections should be saved as tags
        view = dataset.match_tags("selected")

        # If no tagged samples, show available tags
        if len(view) == 0:
            all_tags = dataset.distinct("tags")
            if all_tags:
                print(
                    f"\nNo samples tagged with 'selected'. Available tags: {all_tags}"
                )
            else:
                print("\nNo samples are tagged with 'selected'.")

            print("\nWarning: No samples are tagged for export!")
            print("Please tag samples in the FiftyOne UI before running this script.")
            print("\nCuration workflow:")
            print("  1. Review images in FiftyOne UI")
            print("  2. Select images you want to keep")
            print("  3. Add 'selected' tag to chosen images (via Tags button)")
            print("  4. Run this script to export images with 'selected' tag")
            print("\nAlternatively, export all samples with: --selection-mode all")
            return
    elif selection_mode == "all":
        view = dataset
    elif selection_mode == "with_annotations":
        view = dataset.match(fo.ViewField("has_annotation"))
    else:
        print(f"Error: Unknown selection mode: {selection_mode}")
        return

    print(f"Exporting {len(view)} samples...")

    # Get project root for relative path conversion
    project_root = get_project_root()

    # Prepare data structures
    images_data = []

    for sample in view:
        # Get file paths
        image_path = Path(sample.filepath)

        # Construct label path (replace images/ with labels/ and .jpg with .txt)
        label_path = Path(str(image_path).replace("/images/", "/labels/")).with_suffix(
            ".txt"
        )

        # Convert to relative paths
        image_path_rel = to_relative_path(
            absolute_path=image_path, project_root=project_root
        )
        label_path_rel = (
            to_relative_path(absolute_path=label_path, project_root=project_root)
            if sample.has_annotation
            else None
        )

        # Prepare JSON entry with full metadata
        image_entry = {
            "image_path": image_path_rel,
            "label_path": label_path_rel,
            "location": sample.location,
            "individual": sample.individual,
            "body_part": sample.body_part,
            "has_annotation": sample.has_annotation,
            "sam_score": sample.sam_score,
            "gdino_score": sample.gdino_score,
            "polygon_points": sample.polygon_points,
        }
        images_data.append(image_entry)

    # Calculate statistics
    total_count = len(dataset)
    selected_count = len(view)
    with_annotations = sum(1 for s in view if s.has_annotation)
    without_annotations = selected_count - with_annotations

    # Prepare JSON output
    json_output = {
        "dataset_name": dataset_name,
        "selection_mode": selection_mode,
        "statistics": {
            "total_dataset_count": total_count,
            "selected_count": selected_count,
            "with_annotations": with_annotations,
            "without_annotations": without_annotations,
            "selection_percentage": round(selected_count / total_count * 100, 2)
            if total_count > 0
            else 0,
        },
        "selected_samples": images_data,
    }

    # Create output directory if needed
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON
    with open(output_json, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"\nJSON exported to: {output_json}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("EXPORT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"Selection mode: {selection_mode}")
    print(f"\nTotal dataset: {total_count} images")
    print(
        f"Selected: {selected_count} images ({json_output['statistics']['selection_percentage']}%)"
    )
    print(f"  - With annotations: {with_annotations}")
    print(f"  - Without annotations: {without_annotations}")

    # Group by location and individual
    location_counts = {}
    individual_counts = {}
    for sample in view:
        location_counts[sample.location] = location_counts.get(sample.location, 0) + 1
        individual_counts[sample.individual] = (
            individual_counts.get(sample.individual, 0) + 1
        )

    print("\nBreakdown by location:")
    for location, count in sorted(location_counts.items()):
        print(f"  - {location}: {count} images")

    print("\nBreakdown by individual:")
    for individual, count in sorted(individual_counts.items()):
        print(f"  - {individual}: {count} images")

    print(f"\n{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Export selected samples from FiftyOne dataset"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="snowleopard_yolo_segmentation",
        help="Name of the FiftyOne dataset (default: snowleopard_yolo_segmentation)",
    )
    parser.add_argument(
        "--dir-save",
        type=Path,
        default=Path("data/02_processed/fiftyone/yolo/segmentation"),
        help="Directory to save output files (default: data/02_processed/fiftyone/yolo/segmentation)",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=["selected", "all", "with_annotations"],
        default="selected",
        help="What to export: 'selected' (checked in UI), 'all', or 'with_annotations'",
    )

    args = parser.parse_args()

    # Construct output path from dir_save
    output_json = args.dir_save / "fiftyone_curated_selection.json"

    export_selection(
        dataset_name=args.dataset_name,
        output_json=output_json,
        selection_mode=args.selection_mode,
    )


if __name__ == "__main__":
    main()
