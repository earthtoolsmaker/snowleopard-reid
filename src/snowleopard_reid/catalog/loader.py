"""Utilities for loading and managing the snow leopard catalog.

This module provides functions for loading catalog metadata, individual leopard
information, and catalog features for matching operations.
"""

from pathlib import Path

import torch
import yaml


def load_catalog_index(catalog_root: Path) -> dict:
    """Load the catalog index YAML file.

    Args:
        catalog_root: Path to catalog root directory (e.g., data/08_catalog/v1.0/)

    Returns:
        Dictionary with catalog index data including:
            - catalog_version: str
            - feature_extractors: dict
            - individuals: list
            - statistics: dict

    Raises:
        FileNotFoundError: If catalog index file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    index_path = catalog_root / "catalog_index.yaml"

    if not index_path.exists():
        raise FileNotFoundError(f"Catalog index not found: {index_path}")

    try:
        with open(index_path) as f:
            index = yaml.safe_load(f)
        return index
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse catalog index: {e}")


def load_leopard_metadata(metadata_path: Path) -> dict:
    """Load metadata YAML file for a specific leopard.

    Args:
        metadata_path: Path to leopard metadata.yaml file

    Returns:
        Dictionary with leopard metadata including:
            - individual_id: str
            - leopard_name: str
            - reference_images: list
            - statistics: dict

    Raises:
        FileNotFoundError: If metadata file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Leopard metadata not found: {metadata_path}")

    try:
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
        return metadata
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse leopard metadata: {e}")


def get_all_catalog_features(
    catalog_root: Path,
    extractor: str = "sift",
) -> dict[str, dict[str, torch.Tensor]]:
    """Load all catalog features for a specific extractor.

    Args:
        catalog_root: Path to catalog root directory (e.g., data/08_catalog/v1.0/)
        extractor: Feature extractor name (default: 'sift')

    Returns:
        Dictionary mapping catalog_id to feature dict:
            {
                "leopard1_2022_001": {
                    "keypoints": torch.Tensor,
                    "descriptors": torch.Tensor,
                    "scores": torch.Tensor,
                    ...
                },
                ...
            }

    Raises:
        FileNotFoundError: If catalog doesn't exist
        ValueError: If no features found for extractor
    """
    if not catalog_root.exists():
        raise FileNotFoundError(f"Catalog root not found: {catalog_root}")

    # Load catalog index to get all individuals
    index = load_catalog_index(catalog_root)

    # Check if extractor is available
    available_extractors = index.get("feature_extractors", {})
    if extractor not in available_extractors:
        raise ValueError(
            f"Extractor '{extractor}' not available in catalog. "
            f"Available: {list(available_extractors.keys())}"
        )

    catalog_features = {}
    database_dir = catalog_root / "database"

    # Load features for each individual
    for individual in index["individuals"]:
        # Support both 'leopard_name' and 'individual_name' keys
        leopard_name = individual.get("leopard_name") or individual.get(
            "individual_name"
        )
        location = individual.get("location", "")

        # Construct path: database/{location}/{individual_name}/
        if location:
            leopard_dir = database_dir / location / leopard_name
        else:
            leopard_dir = database_dir / leopard_name

        # Load leopard metadata to get all reference images
        metadata_path = leopard_dir / "metadata.yaml"
        metadata = load_leopard_metadata(metadata_path)

        # Load features for each reference image
        for ref_image in metadata["reference_images"]:
            # Check if features exist for this extractor
            if extractor not in ref_image.get("features", {}):
                continue

            # Get feature path (relative to database directory in metadata)
            feature_rel_path = ref_image["features"][extractor]
            feature_path = database_dir / feature_rel_path

            if not feature_path.exists():
                # Skip missing features with a warning
                continue

            # Create catalog ID: leopard_name_year_imagenum
            # e.g., "naguima_2022_001"
            image_id = ref_image["image_id"]
            catalog_id = f"{leopard_name.lower().replace(' ', '_')}_{image_id}"

            # Load features
            try:
                feats = torch.load(feature_path, map_location="cpu", weights_only=False)
                catalog_features[catalog_id] = feats
            except Exception:
                # Skip files that can't be loaded
                continue

    if not catalog_features:
        raise ValueError(f"No features found for extractor '{extractor}' in catalog")

    return catalog_features


def get_filtered_catalog_features(
    catalog_root: Path,
    extractor: str = "sift",
    locations: list[str] | None = None,
    body_parts: list[str] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Load filtered catalog features for a specific extractor.

    Args:
        catalog_root: Path to catalog root directory (e.g., data/08_catalog/v1.0/)
        extractor: Feature extractor name (default: 'sift')
        locations: List of locations to filter by (e.g., ["naryn", "sarychat"]).
                   If None, includes all locations.
        body_parts: List of body parts to filter by (e.g., ["head", "right_flank"]).
                    If None, includes all body parts.

    Returns:
        Dictionary mapping catalog_id to feature dict:
            {
                "leopard1_2022_001": {
                    "keypoints": torch.Tensor,
                    "descriptors": torch.Tensor,
                    "scores": torch.Tensor,
                    ...
                },
                ...
            }

    Raises:
        FileNotFoundError: If catalog doesn't exist
        ValueError: If no features found for extractor or filters
    """
    if not catalog_root.exists():
        raise FileNotFoundError(f"Catalog root not found: {catalog_root}")

    # Load catalog index to get all individuals
    index = load_catalog_index(catalog_root)

    # Check if extractor is available
    available_extractors = index.get("feature_extractors", {})
    if extractor not in available_extractors:
        raise ValueError(
            f"Extractor '{extractor}' not available in catalog. "
            f"Available: {list(available_extractors.keys())}"
        )

    catalog_features = {}
    database_dir = catalog_root / "database"

    # Load features for each individual
    for individual in index["individuals"]:
        # Support both 'leopard_name' and 'individual_name' keys
        leopard_name = individual.get("leopard_name") or individual.get(
            "individual_name"
        )
        location = individual.get("location", "")

        # Filter by location if specified
        if locations is not None and location not in locations:
            continue

        # Construct path: database/{location}/{individual_name}/
        if location:
            leopard_dir = database_dir / location / leopard_name
        else:
            leopard_dir = database_dir / leopard_name

        # Load leopard metadata to get all reference images
        metadata_path = leopard_dir / "metadata.yaml"
        metadata = load_leopard_metadata(metadata_path)

        # Load features for each reference image
        for ref_image in metadata["reference_images"]:
            # Filter by body part if specified
            if body_parts is not None:
                ref_body_part = ref_image.get("body_part", "")
                if ref_body_part not in body_parts:
                    continue

            # Check if features exist for this extractor
            if extractor not in ref_image.get("features", {}):
                continue

            # Get feature path (relative to database directory in metadata)
            feature_rel_path = ref_image["features"][extractor]
            feature_path = database_dir / feature_rel_path

            if not feature_path.exists():
                # Skip missing features with a warning
                continue

            # Create catalog ID: leopard_name_year_imagenum
            # e.g., "naguima_2022_001"
            image_id = ref_image["image_id"]
            catalog_id = f"{leopard_name.lower().replace(' ', '_')}_{image_id}"

            # Load features
            try:
                feats = torch.load(feature_path, map_location="cpu", weights_only=False)
                catalog_features[catalog_id] = feats
            except Exception:
                # Skip files that can't be loaded
                continue

    if not catalog_features:
        filter_info = []
        if locations:
            filter_info.append(f"locations={locations}")
        if body_parts:
            filter_info.append(f"body_parts={body_parts}")
        filter_str = ", ".join(filter_info) if filter_info else "no filters"
        raise ValueError(
            f"No features found for extractor '{extractor}' with {filter_str}"
        )

    return catalog_features


def get_available_locations(catalog_root: Path) -> list[str]:
    """Get list of available locations from catalog.

    Args:
        catalog_root: Path to catalog root directory

    Returns:
        List of location names prepended with "all" (e.g., ["all", "naryn", "sarychat"])
    """
    try:
        index = load_catalog_index(catalog_root)
        locations = index.get("statistics", {}).get("locations", [])
        return ["all"] + sorted(locations)
    except Exception:
        return ["all"]


def get_available_body_parts(catalog_root: Path) -> list[str]:
    """Get list of available body parts from catalog.

    Args:
        catalog_root: Path to catalog root directory

    Returns:
        List of body part names prepended with "all"
        (e.g., ["all", "head", "left_flank", "right_flank", "tail", "misc"])
    """
    try:
        index = load_catalog_index(catalog_root)
        body_parts = index.get("statistics", {}).get("body_parts", [])
        return ["all"] + sorted(body_parts)
    except Exception:
        return ["all"]


def get_catalog_metadata_for_id(
    catalog_root: Path,
    catalog_id: str,
) -> dict | None:
    """Get full metadata for a specific catalog ID.

    Args:
        catalog_root: Path to catalog root directory
        catalog_id: Catalog ID (e.g., "naguima_2022_001")

    Returns:
        Dictionary with metadata including:
            - leopard_name: str
            - year: int
            - image_path: Path
            - individual_id: str
        Or None if not found

    Raises:
        FileNotFoundError: If catalog doesn't exist
    """
    if not catalog_root.exists():
        raise FileNotFoundError(f"Catalog root not found: {catalog_root}")

    # Load catalog index
    index = load_catalog_index(catalog_root)
    database_dir = catalog_root / "database"

    # Try to find matching individual
    for individual in index["individuals"]:
        # Support both 'leopard_name' and 'individual_name' keys
        leopard_name = individual.get("leopard_name") or individual.get(
            "individual_name"
        )
        location = individual.get("location", "")

        # Construct path: database/{location}/{individual_name}/
        if location:
            leopard_dir = database_dir / location / leopard_name
        else:
            leopard_dir = database_dir / leopard_name

        # Load leopard metadata
        metadata_path = leopard_dir / "metadata.yaml"
        metadata = load_leopard_metadata(metadata_path)

        # Check each reference image
        for ref_image in metadata["reference_images"]:
            # Construct expected catalog ID
            image_id = ref_image["image_id"]
            expected_id = f"{leopard_name.lower().replace(' ', '_')}_{image_id}"

            if expected_id == catalog_id:
                # Found match
                return {
                    "leopard_name": leopard_name,
                    "image_path": database_dir / ref_image["path"],
                    "individual_id": metadata["individual_id"],
                    "filename": ref_image["filename"],
                }

    return None
