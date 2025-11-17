"""Generate metadata files for snow leopard catalog.

This script creates comprehensive metadata for the catalog, generating both individual
leopard metadata files and a global catalog index with statistics.

This script:
1. Scans catalog database for all individual directories
2. Counts images and feature files for each individual (organized by location/body_part)
3. Generates individual metadata YAML files (individual_name/metadata.yaml)
4. Creates catalog index YAML with aggregate statistics
5. Logs catalog summary and any inconsistencies

Usage:
    python scripts/catalog/generate_catalog_metadata.py --catalog-dir <path> [options]

Example:
    python scripts/catalog/generate_catalog_metadata.py \
        --catalog-dir ./data/08_catalog/v1.0 \
        --catalog-version 1.0.0

Notes:
    - Run after build_catalog_from_selection.py and extract_catalog_features.py
    - Catalog index saved to catalog_dir/catalog_index.yaml
    - Individual metadata saved to database/individual_name/metadata.yaml
"""

import argparse
import logging
from pathlib import Path

import torch
import yaml


def str_presenter(dumper, data):
    """Custom YAML string representer that quotes strings containing spaces."""
    if " " in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def setup_logging(verbose: bool) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def generate_individual_metadata(
    individual_dir: Path, catalog_root: Path, available_extractors: list[str]
) -> dict:
    """Generate metadata for a single individual.

    Args:
        individual_dir: Path to individual directory
        catalog_root: Root catalog database directory
        available_extractors: List of feature extractors that have been run

    Returns:
        Metadata dictionary
    """
    individual_name = individual_dir.name
    images_dir = individual_dir / "images"
    features_dir = individual_dir / "features"

    # Find all images
    image_files = list(images_dir.rglob("*.jpg"))

    reference_images = []
    locations_set = set()
    body_parts_set = set()
    total_keypoints_by_extractor = {ext: 0 for ext in available_extractors}

    # Track images per location and body part
    images_per_location = {}
    images_per_body_part = {}

    # Extract location from individual_dir path (database/{location}/{individual}/)
    location = individual_dir.parent.name

    for image_path in sorted(image_files):
        # Extract body_part from path structure: images/{body_part}/filename.jpg
        relative_path = image_path.relative_to(images_dir)
        parts = relative_path.parts

        if len(parts) >= 1:
            body_part = parts[0]
        else:
            logging.warning(f"Unexpected path structure for {image_path}, skipping")
            continue

        locations_set.add(location)
        body_parts_set.add(body_part)

        # Count images per location and body part
        images_per_location[location] = images_per_location.get(location, 0) + 1
        images_per_body_part[body_part] = images_per_body_part.get(body_part, 0) + 1

        # Get relative paths from catalog root
        image_rel_path = image_path.relative_to(catalog_root)
        filename = image_path.name
        image_id = filename.replace(".jpg", "")

        # Build features dict
        features_dict = {}
        num_keypoints = 0

        for extractor in available_extractors:
            # Features follow structure: features/{extractor}/{body_part}/filename.pt
            feature_path = (
                features_dir
                / extractor
                / body_part
                / filename.replace(".jpg", ".pt")
            )
            if feature_path.exists():
                # Load feature file to get keypoint count
                try:
                    feats = torch.load(
                        feature_path, map_location="cpu", weights_only=True
                    )
                    kpt_count = feats["keypoints"].shape[0]
                    features_dict[extractor] = str(
                        feature_path.relative_to(catalog_root)
                    )
                    if (
                        extractor == available_extractors[0]
                    ):  # Use first extractor for count
                        num_keypoints = kpt_count
                    total_keypoints_by_extractor[extractor] += kpt_count
                except Exception as e:
                    logging.warning(f"Failed to load features from {feature_path}: {e}")

        image_entry = {
            "image_id": image_id,
            "filename": filename,
            "path": str(image_rel_path),
            "location": location,
            "body_part": body_part,
            "features": features_dict,
        }

        # Add keypoint count if available
        if num_keypoints > 0:
            image_entry["num_keypoints"] = num_keypoints

        reference_images.append(image_entry)

    # Create individual ID (lowercase, replace spaces with underscores)
    individual_id = individual_name.lower().replace(" ", "_") + "_001"

    # Generate statistics
    statistics = {
        "total_reference_images": len(reference_images),
        "locations_represented": sorted(locations_set),
        "body_parts_represented": sorted(body_parts_set),
        "images_per_location": dict(sorted(images_per_location.items())),
        "images_per_body_part": dict(sorted(images_per_body_part.items())),
    }

    # Add keypoint stats for each extractor
    for extractor, total_kpts in total_keypoints_by_extractor.items():
        if total_kpts > 0:
            statistics[f"total_keypoints_{extractor}"] = total_kpts

    metadata = {
        "individual_id": individual_id,
        "individual_name": individual_name,
        "location": location,  # Add location field
        "reference_images": reference_images,
        "statistics": statistics,
    }

    return metadata


def generate_catalog_index(
    catalog_dir: Path, individual_metadatas: list[dict], catalog_version: str
) -> dict:
    """Generate catalog index with aggregate statistics.

    Args:
        catalog_dir: Path to catalog database directory
        individual_metadatas: List of individual metadata dicts
        catalog_version: Catalog version string

    Returns:
        Catalog index dictionary
    """
    # Aggregate statistics
    total_images = sum(
        m["statistics"]["total_reference_images"] for m in individual_metadatas
    )

    # Collect all locations and body parts
    all_locations = set()
    all_body_parts = set()
    for metadata in individual_metadatas:
        all_locations.update(metadata["statistics"]["locations_represented"])
        all_body_parts.update(metadata["statistics"]["body_parts_represented"])

    # Calculate images per individual statistics
    images_per_individual = [
        m["statistics"]["total_reference_images"] for m in individual_metadatas
    ]

    statistics = {
        "total_individuals": len(individual_metadatas),
        "total_reference_images": total_images,
        "locations": sorted(all_locations),
        "body_parts": sorted(all_body_parts),
        "images_per_individual": {
            "mean": sum(images_per_individual) / len(images_per_individual)
            if images_per_individual
            else 0,
            "min": min(images_per_individual) if images_per_individual else 0,
            "max": max(images_per_individual) if images_per_individual else 0,
        },
    }

    # Add total keypoints if available
    all_extractors = set()
    for metadata in individual_metadatas:
        for key in metadata["statistics"]:
            if key.startswith("total_keypoints_"):
                all_extractors.add(key.replace("total_keypoints_", ""))

    for extractor in sorted(all_extractors):
        key = f"total_keypoints_{extractor}"
        total_kpts = sum(m["statistics"].get(key, 0) for m in individual_metadatas)
        if total_kpts > 0:
            statistics[key] = total_kpts

    # Build individual list
    individuals = []
    for metadata in sorted(individual_metadatas, key=lambda m: m["individual_name"]):
        location = metadata["location"]
        individual_entry = {
            "individual_id": metadata["individual_id"],
            "individual_name": metadata["individual_name"],
            "location": location,
            "reference_count": metadata["statistics"]["total_reference_images"],
            "locations": metadata["statistics"]["locations_represented"],
            "body_parts": metadata["statistics"]["body_parts_represented"],
            "metadata_path": f"database/{location}/{metadata['individual_name']}/metadata.yaml",
        }
        individuals.append(individual_entry)

    # Detect available feature extractors
    feature_extractors = {}
    if all_extractors:
        for extractor in sorted(all_extractors):
            feature_extractors[extractor] = {
                "backend": "opencv" if extractor == "sift" else "unknown",
                "rootsift": True if extractor == "sift" else False,
            }

    catalog_index = {
        "catalog_version": catalog_version,
        "feature_extractors": feature_extractors,
        "individuals": individuals,
        "statistics": statistics,
    }

    return catalog_index


def generate_catalog_metadata(
    catalog_dir: Path,
    catalog_version: str = "1.0.0",
    verbose: bool = False,
) -> None:
    """Generate all catalog metadata files.

    Args:
        catalog_dir: Path to catalog directory (contains database/ subdirectory)
        catalog_version: Version string for catalog
        verbose: Enable verbose logging
    """
    setup_logging(verbose)

    database_dir = catalog_dir / "database"
    if not database_dir.exists():
        raise FileNotFoundError(f"Database directory not found: {database_dir}")

    logging.info(f"Generating metadata for catalog: {catalog_dir}")

    # Find all location directories first
    location_dirs = [d for d in database_dir.iterdir() if d.is_dir()]
    logging.info(f"Found {len(location_dirs)} location directories")

    # Collect all individuals across locations
    all_individual_dirs = []
    for location_dir in location_dirs:
        individual_dirs = [d for d in location_dir.iterdir() if d.is_dir()]
        all_individual_dirs.extend(individual_dirs)

    logging.info(f"Found {len(all_individual_dirs)} individual directories across all locations")

    # Detect available feature extractors
    available_extractors = []
    if all_individual_dirs:
        features_dir = all_individual_dirs[0] / "features"
        if features_dir.exists():
            available_extractors = [
                d.name for d in features_dir.iterdir() if d.is_dir()
            ]
    logging.info(f"Available feature extractors: {available_extractors}")

    # Generate metadata for each individual
    individual_metadatas = []
    for individual_dir in sorted(all_individual_dirs):
        try:
            metadata = generate_individual_metadata(
                individual_dir, database_dir, available_extractors
            )
            individual_metadatas.append(metadata)

            # Save individual metadata with custom string representer for quoting
            yaml.add_representer(str, str_presenter)
            metadata_path = individual_dir / "metadata.yaml"
            with open(metadata_path, "w") as f:
                yaml.dump(
                    metadata, f, default_flow_style=False, sort_keys=False, width=1000
                )
            location_name = individual_dir.parent.name
            logging.info(
                f"Generated metadata for {location_name}/{metadata['individual_name']}: "
                f"{metadata['statistics']['total_reference_images']} images"
            )
        except Exception as e:
            logging.error(f"Failed to generate metadata for {individual_dir.name}: {e}")

    # Generate catalog index
    catalog_index = generate_catalog_index(
        database_dir, individual_metadatas, catalog_version
    )

    # Save catalog index with custom string representer for quoting
    yaml.add_representer(str, str_presenter)
    index_path = catalog_dir / "catalog_index.yaml"
    with open(index_path, "w") as f:
        yaml.dump(
            catalog_index, f, default_flow_style=False, sort_keys=False, width=1000
        )

    logging.info("\nCatalog metadata generation complete!")
    logging.info(
        f"  Total individuals: {catalog_index['statistics']['total_individuals']}"
    )
    logging.info(
        f"  Total images: {catalog_index['statistics']['total_reference_images']}"
    )
    logging.info(f"  Locations: {', '.join(catalog_index['statistics']['locations'])}")
    logging.info(
        f"  Body parts: {', '.join(catalog_index['statistics']['body_parts'])}"
    )
    logging.info(f"\nCatalog index saved to: {index_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate catalog metadata files")
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        required=True,
        help="Path to catalog directory (contains database/ subdirectory)",
    )
    parser.add_argument(
        "--catalog-version",
        type=str,
        default="1.0.0",
        help="Version string for catalog (default: 1.0.0)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    generate_catalog_metadata(
        catalog_dir=args.catalog_dir,
        catalog_version=args.catalog_version,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
