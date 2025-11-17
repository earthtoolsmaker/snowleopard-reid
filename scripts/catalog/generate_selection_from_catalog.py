"""Generate selection.yaml from existing catalog structure.

This script scans the curated catalog in data/02_processed/catalog/ and generates
a selection.yaml file that maps to the original SAM cropped images.

The output format is:
selections:
  {location}:
    {individual}:
      {body_part}:
        - path/to/image1.jpg
        - path/to/image2.jpg
"""

import argparse
from pathlib import Path

import yaml


def str_presenter(dumper, data):
    """Custom YAML string representer that quotes strings containing spaces."""
    if " " in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def scan_catalog_directory(catalog_dir: Path, cropped_dir: Path) -> dict:
    """Scan catalog directory and map to cropped leopard images.

    Args:
        catalog_dir: Path to data/02_processed/catalog/
        cropped_dir: Path to data/02_processed/cropped/leopards/

    Returns:
        Dictionary with structure: {location: {individual: {body_part: [paths]}}}
    """
    selections = {}

    # Iterate through locations
    for location_dir in sorted(catalog_dir.iterdir()):
        if not location_dir.is_dir():
            continue

        location = location_dir.name
        selections[location] = {}

        # Iterate through individuals in this location
        for individual_dir in sorted(location_dir.iterdir()):
            if not individual_dir.is_dir():
                continue

            individual = individual_dir.name
            selections[location][individual] = {}

            # Iterate through body parts
            for body_part_dir in sorted(individual_dir.iterdir()):
                if not body_part_dir.is_dir():
                    continue

                body_part = body_part_dir.name
                image_paths = []

                # Collect all images in this body part directory
                for image_file in sorted(body_part_dir.glob("*.jpg")):
                    # Construct path to original cropped image
                    cropped_path = (
                        cropped_dir
                        / location
                        / individual
                        / body_part
                        / image_file.name
                    )

                    # Verify the image exists in the cropped directory
                    if cropped_path.exists():
                        # Store as string path (already relative if input was relative)
                        image_paths.append(str(cropped_path))
                    else:
                        # Try to find the image in other body part folders
                        found = False
                        individual_dir = cropped_dir / location / individual
                        if individual_dir.exists():
                            for alt_body_part in individual_dir.iterdir():
                                if alt_body_part.is_dir():
                                    alt_path = alt_body_part / image_file.name
                                    if alt_path.exists():
                                        print(
                                            f"Info: Found {image_file.name} in {alt_body_part.name}/ instead of {body_part}/"
                                        )
                                        image_paths.append(str(alt_path))
                                        found = True
                                        break

                        if not found:
                            print(
                                f"Warning: Could not find {image_file.name} for {individual} in any body part folder"
                            )

                # Only add body part if it has images
                if image_paths:
                    selections[location][individual][body_part] = image_paths

            # Remove individual if no body parts with images
            if not selections[location][individual]:
                del selections[location][individual]

        # Remove location if no individuals
        if not selections[location]:
            del selections[location]

    return {"selections": selections}


def main():
    parser = argparse.ArgumentParser(
        description="Generate selection.yaml from catalog directory structure"
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=Path("data/02_processed/catalog"),
        help="Path to curated catalog directory (default: data/02_processed/catalog)",
    )
    parser.add_argument(
        "--cropped-dir",
        type=Path,
        default=Path("data/02_processed/cropped/leopards"),
        help="Path to SAM cropped images (default: data/02_processed/cropped/leopards)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/08_catalog/v1.0/selection.yaml"),
        help="Output path for selection.yaml (default: data/08_catalog/v1.0/selection.yaml)",
    )

    args = parser.parse_args()

    # Validate input directories
    if not args.catalog_dir.exists():
        raise FileNotFoundError(f"Catalog directory not found: {args.catalog_dir}")

    if not args.cropped_dir.exists():
        raise FileNotFoundError(f"Cropped directory not found: {args.cropped_dir}")

    print(f"Scanning catalog directory: {args.catalog_dir}")
    print(f"Mapping to cropped images in: {args.cropped_dir}")

    # Scan and generate selections
    selections_data = scan_catalog_directory(args.catalog_dir, args.cropped_dir)

    # Print statistics
    total_individuals = sum(
        len(individuals) for individuals in selections_data["selections"].values()
    )
    total_images = sum(
        len(images)
        for location_data in selections_data["selections"].values()
        for individual_data in location_data.values()
        for images in individual_data.values()
    )

    print("\nFound:")
    print(f"  - {len(selections_data['selections'])} locations")
    print(f"  - {total_individuals} individuals")
    print(f"  - {total_images} images")

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write selection.yaml with custom string representer for quoting paths with spaces
    yaml.add_representer(str, str_presenter)
    with open(args.output, "w") as f:
        yaml.dump(
            selections_data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=1000,  # Prevent line wrapping for long paths
        )

    print(f"\nSelection file written to: {args.output}")


if __name__ == "__main__":
    main()
