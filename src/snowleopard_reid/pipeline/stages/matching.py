"""Matching stage for snow leopard identification.

This module handles matching query features against the catalog using
LightGlue and computing matching metrics.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightglue import LightGlue
from scipy.stats import wasserstein_distance

from snowleopard_reid import get_device
from snowleopard_reid.catalog import (
    get_all_catalog_features,
    get_catalog_metadata_for_id,
    load_catalog_index,
)

logger = logging.getLogger(__name__)


def run_matching_stage(
    query_features: dict[str, torch.Tensor],
    catalog_path: Path | str,
    top_k: int = 5,
    extractor: str = "sift",
    device: str | None = None,
    query_image_path: str | None = None,
    pairwise_output_dir: Path | None = None,
) -> dict:
    """Match query against catalog.

    This stage matches the query features against all catalog images using
    LightGlue, computes metrics, and ranks matches.

    Args:
        query_features: Query features dict with keypoints, descriptors, scores
        catalog_path: Path to catalog root directory (e.g., data/08_catalog/v1.0/)
        top_k: Number of top matches to return (default: 5)
        extractor: Feature extractor used (default: 'sift')
        device: Device to run matching on ('cpu', 'cuda', or None for auto-detect)
        query_image_path: Path to query image (optional, for pairwise data)
        pairwise_output_dir: Directory to save pairwise match data (optional)

    Returns:
        Stage dict with structure:
        {
            "stage_id": "matching",
            "stage_name": "Matching",
            "description": "Match query against catalog using LightGlue",
            "config": {...},
            "metrics": {...},
            "data": {
                "catalog_info": {...},
                "matches": [...]
            }
        }

    Raises:
        FileNotFoundError: If catalog doesn't exist
        ValueError: If extractor not available in catalog
        RuntimeError: If matching fails
    """
    catalog_path = Path(catalog_path)

    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    # Auto-detect device
    device = get_device(device=device, verbose=True)

    # Load catalog index
    logger.info(f"Loading catalog from {catalog_path}")
    catalog_index = load_catalog_index(catalog_path)
    logger.info(
        f"Catalog v{catalog_index['catalog_version']}: "
        f"{catalog_index['statistics']['total_individuals']} individuals, "
        f"{catalog_index['statistics']['total_reference_images']} images"
    )

    # Load all catalog features
    logger.info(f"Loading catalog features (extractor: {extractor})")
    try:
        catalog_features = get_all_catalog_features(
            catalog_root=catalog_path, extractor=extractor
        )
        logger.info(f"Loaded {len(catalog_features)} catalog features")
    except ValueError as e:
        raise ValueError(f"Failed to load catalog features: {e}")

    # Initialize LightGlue matcher
    logger.info(f"Initializing LightGlue matcher with {extractor} features")
    try:
        matcher = LightGlue(features=extractor).eval().to(device)
    except Exception as e:
        raise ValueError(
            f"Failed to initialize LightGlue matcher with extractor '{extractor}': {e}"
        )

    # Move query features to device and add batch dimension
    query_feats = {}
    for k, v in query_features.items():
        if isinstance(v, torch.Tensor):
            # Add batch dimension if not present
            if v.ndim == 1:
                v = v.unsqueeze(0)
            elif v.ndim == 2:
                v = v.unsqueeze(0)
            query_feats[k] = v.to(device)
        else:
            query_feats[k] = v

    # Serial matching: iterate through catalog
    logger.info(f"Matching against {len(catalog_features)} catalog images")
    matches_dict = {}
    raw_matches_cache = {}  # Store raw matches for pairwise saving

    for catalog_id, catalog_feats in catalog_features.items():
        # Move catalog features to device and add batch dimension
        catalog_feats_device = {}
        for k, v in catalog_feats.items():
            if isinstance(v, torch.Tensor):
                # Add batch dimension if not present
                if v.ndim == 1:
                    v = v.unsqueeze(0)
                elif v.ndim == 2:
                    v = v.unsqueeze(0)
                catalog_feats_device[k] = v.to(device)
            else:
                catalog_feats_device[k] = v

        # Run matcher
        try:
            with torch.no_grad():
                matches = matcher(
                    {
                        "image0": query_feats,
                        "image1": catalog_feats_device,
                    }
                )
        except Exception as e:
            logger.warning(f"Matching failed for {catalog_id}: {e}")
            continue

        # Compute metrics
        try:
            metrics = compute_match_metrics(matches)
            matches_dict[catalog_id] = metrics

            # Cache raw matches and features for top-k pairwise saving
            if pairwise_output_dir is not None:
                raw_matches_cache[catalog_id] = {
                    "matches": matches,
                    "catalog_features": catalog_feats,
                }
        except KeyError as e:
            logger.warning(f"Failed to compute metrics for {catalog_id}: {e}")
            continue

    logger.info(f"Successfully matched against {len(matches_dict)} catalog images")

    if not matches_dict:
        raise RuntimeError(
            "No successful matches found. All catalog images failed to match. "
            "This may indicate a problem with feature extraction or format."
        )

    # Rank matches by Wasserstein distance
    ranked_matches = rank_matches(matches_dict, metric="wasserstein", top_k=top_k)

    # Enrich matches with catalog metadata
    enriched_matches = []
    for match in ranked_matches:
        catalog_id = match["catalog_id"]
        metadata = get_catalog_metadata_for_id(
            catalog_root=catalog_path, catalog_id=catalog_id
        )

        if metadata is None:
            logger.warning(f"No metadata found for {catalog_id}")
            continue

        enriched_match = {
            "rank": match["rank"],
            "catalog_id": catalog_id,
            "leopard_name": metadata["leopard_name"],
            "filepath": str(metadata["image_path"]),
            "wasserstein": match["wasserstein"],
            "auc": match["auc"],
            "num_matches": match["num_matches"],
            "individual_id": metadata["individual_id"],
        }
        enriched_matches.append(enriched_match)

    if enriched_matches:
        logger.info(
            f"Top match: {enriched_matches[0]['leopard_name']} "
            f"(wasserstein: {enriched_matches[0]['wasserstein']:.4f}, "
            f"matches: {enriched_matches[0]['num_matches']})"
        )

    # Save pairwise match data for top-k matches
    if pairwise_output_dir is not None and enriched_matches:
        pairwise_output_dir = Path(pairwise_output_dir)
        pairwise_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Saving pairwise match data for top-{len(enriched_matches)} matches"
        )

        # Get query image size
        query_image_size = query_features.get("image_size")
        if isinstance(query_image_size, torch.Tensor):
            query_image_size = query_image_size.cpu().numpy()

        for enriched_match in enriched_matches:
            catalog_id = enriched_match["catalog_id"]

            # Skip if no cached data for this catalog_id
            if catalog_id not in raw_matches_cache:
                logger.warning(
                    f"No cached match data for {catalog_id}, skipping pairwise save"
                )
                enriched_match["pairwise_file"] = None
                continue

            # Get cached data
            cached = raw_matches_cache[catalog_id]
            matches = cached["matches"]
            catalog_feats = cached["catalog_features"]

            # Extract matched keypoints
            try:
                matched_data = extract_matched_keypoints(
                    query_features=query_features,
                    catalog_features=catalog_feats,
                    matches=matches,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to extract keypoints for {catalog_id}: {e}, skipping"
                )
                enriched_match["pairwise_file"] = None
                continue

            # Get catalog image size
            catalog_image_size = catalog_feats.get("image_size")
            if isinstance(catalog_image_size, torch.Tensor):
                catalog_image_size = catalog_image_size.cpu().numpy()

            # Build pairwise data
            pairwise_data = {
                "rank": enriched_match["rank"],
                "catalog_id": catalog_id,
                "leopard_name": enriched_match["leopard_name"],
                "query_image_path": query_image_path or "",
                "catalog_image_path": enriched_match["filepath"],
                "query_image_size": query_image_size,
                "catalog_image_size": catalog_image_size,
                "query_keypoints": matched_data["query_keypoints"],
                "catalog_keypoints": matched_data["catalog_keypoints"],
                "match_scores": matched_data["match_scores"],
                "wasserstein": enriched_match["wasserstein"],
                "auc": enriched_match["auc"],
                "num_matches": matched_data[
                    "num_matches"
                ],  # Use actual count from extracted keypoints
            }

            # Save as compressed NPZ
            output_filename = f"rank_{enriched_match['rank']:02d}_{catalog_id}.npz"
            output_path = pairwise_output_dir / output_filename

            np.savez_compressed(output_path, **pairwise_data)

            # Add pairwise file reference to enriched_match (relative to matching stage dir)
            enriched_match["pairwise_file"] = f"pairwise/{output_filename}"

        logger.info(f"Saved pairwise data to {pairwise_output_dir}")
    else:
        # Set pairwise_file to None if not saving pairwise data
        for enriched_match in enriched_matches:
            enriched_match["pairwise_file"] = None

    # Return standardized stage dict
    return {
        "stage_id": "matching",
        "stage_name": "Matching",
        "description": "Match query against catalog using LightGlue",
        "config": {
            "top_k": top_k,
            "extractor": extractor,
            "device": device,
            "catalog_path": str(catalog_path),
        },
        "metrics": {
            "num_catalog_images": len(catalog_features),
            "num_successful_matches": len(matches_dict),
            "top_match_wasserstein": enriched_matches[0]["wasserstein"]
            if enriched_matches
            else 0.0,
            "top_match_leopard_name": enriched_matches[0]["leopard_name"]
            if enriched_matches
            else "",
        },
        "data": {
            "catalog_info": {
                "catalog_version": catalog_index["catalog_version"],
                "catalog_path": str(catalog_path),
                "num_individuals": catalog_index["statistics"]["total_individuals"],
                "num_reference_images": catalog_index["statistics"][
                    "total_reference_images"
                ],
            },
            "matches": enriched_matches,
        },
    }


# ============================================================================
# Metrics Utilities
# ============================================================================


def compute_wasserstein_distance(scores: np.ndarray) -> float:
    """Compute Wasserstein distance from null distribution.

    The Wasserstein distance measures how far the match score distribution is from
    a null distribution (all zeros). Higher values indicate better matches.
    This is the optimal metric for re-identification tasks.

    Args:
        scores: Array of match scores (typically from matcher output)

    Returns:
        Wasserstein distance as a float

    References:
        Based on trout-reid implementation for animal re-identification
    """
    if len(scores) == 0:
        return 0.0

    # Null distribution: fixed-length array of zeros
    # This represents no matches at all
    # Using fixed length (1024) ensures all matches are comparable
    # to the same reference distribution (follows trout-reID implementation)
    x_null_distribution = np.zeros(1024)

    # Compute Wasserstein (Earth Mover's) distance
    distance = wasserstein_distance(x_null_distribution, scores)

    return float(distance)


def compute_auc(scores: np.ndarray) -> float:
    """Compute Area Under Curve (cumulative distribution) of match scores.

    AUC represents the cumulative distribution of match scores.
    Higher values indicate better matches.

    Args:
        scores: Array of match scores (typically from matcher output)

    Returns:
        AUC value as a float (0.0 to 1.0)

    References:
        Based on trout-reid implementation
    """
    if len(scores) == 0:
        return 0.0

    # Sort scores in ascending order
    sorted_scores = np.sort(scores)

    # Compute cumulative sum
    cumsum = np.cumsum(sorted_scores)

    # Normalize by total sum to get AUC in [0, 1]
    if cumsum[-1] > 0:
        auc = np.trapz(cumsum / cumsum[-1]) / len(scores)
    else:
        auc = 0.0

    return float(auc)


def extract_match_scores(matches: dict[str, torch.Tensor]) -> np.ndarray:
    """Extract match scores from matcher output.

    Args:
        matches: Dictionary from LightGlue matcher with keys:
            - matches0: Tensor of matched indices
            - matching_scores0: Tensor of match confidence scores

    Returns:
        Numpy array of match scores

    Raises:
        KeyError: If required keys are missing from matches dict
    """
    if "matching_scores0" not in matches:
        raise KeyError("matches dictionary missing 'matching_scores0' key")

    scores = matches["matching_scores0"]

    # Convert to numpy and filter out invalid matches (-1 values)
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    # Filter out unmatched keypoints (score = 0 or negative)
    valid_scores = scores[scores > 0]

    return valid_scores


def extract_matched_keypoints(
    query_features: dict[str, torch.Tensor],
    catalog_features: dict[str, torch.Tensor],
    matches: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    """Extract matched keypoint pairs from matcher output.

    Args:
        query_features: Query feature dict with 'keypoints' tensor [M, 2]
        catalog_features: Catalog feature dict with 'keypoints' tensor [N, 2]
        matches: Dictionary from LightGlue matcher with:
            - matches0: Tensor [M] mapping query_idx → catalog_idx (-1 if no match)
            - matching_scores0: Tensor [M] with match confidence scores

    Returns:
        Dictionary with:
            - query_keypoints: ndarray [num_matches, 2] - matched query keypoints
            - catalog_keypoints: ndarray [num_matches, 2] - matched catalog keypoints
            - match_scores: ndarray [num_matches] - confidence scores
            - num_matches: int - number of valid matches

    Raises:
        KeyError: If required keys are missing
    """
    if "matches0" not in matches or "matching_scores0" not in matches:
        raise KeyError(
            "matches dictionary missing 'matches0' or 'matching_scores0' keys"
        )

    # Get match indices and scores
    matches0 = matches["matches0"]  # Shape: [M]
    scores0 = matches["matching_scores0"]  # Shape: [M]

    # Convert to numpy if tensors
    if isinstance(matches0, torch.Tensor):
        matches0 = matches0.cpu().numpy()
    if isinstance(scores0, torch.Tensor):
        scores0 = scores0.cpu().numpy()

    # Remove batch dimension if present
    if matches0.ndim == 2:
        matches0 = matches0[0]
    if scores0.ndim == 2:
        scores0 = scores0[0]

    # Filter valid matches (matched and score > 0)
    valid_mask = (matches0 >= 0) & (scores0 > 0)
    valid_indices = matches0[valid_mask].astype(int)
    valid_scores = scores0[valid_mask]

    # Get keypoints
    query_kpts = query_features["keypoints"]
    catalog_kpts = catalog_features["keypoints"]

    # Convert to numpy if tensors
    if isinstance(query_kpts, torch.Tensor):
        query_kpts = query_kpts.cpu().numpy()
    if isinstance(catalog_kpts, torch.Tensor):
        catalog_kpts = catalog_kpts.cpu().numpy()

    # Remove batch dimension if present
    if query_kpts.ndim == 3:
        query_kpts = query_kpts[0]
    if catalog_kpts.ndim == 3:
        catalog_kpts = catalog_kpts[0]

    # Extract matched keypoints
    query_matched = query_kpts[valid_mask]
    catalog_matched = catalog_kpts[valid_indices]

    return {
        "query_keypoints": query_matched,
        "catalog_keypoints": catalog_matched,
        "match_scores": valid_scores,
        "num_matches": len(valid_scores),
    }


def rank_matches(
    matches_dict: dict[str, dict[str, Any]],
    metric: str = "wasserstein",
    top_k: int = None,
) -> list[dict[str, Any]]:
    """Rank matches by specified metric.

    Args:
        matches_dict: Dictionary mapping catalog_id to match info
        metric: Metric to rank by ('wasserstein' or 'auc')
        top_k: Number of top matches to return (None = all)

    Returns:
        List of match dictionaries sorted by metric (best first)

    Raises:
        ValueError: If metric is not supported
    """
    if metric not in ["wasserstein", "auc"]:
        raise ValueError(f"Unsupported metric: {metric}. Use 'wasserstein' or 'auc'")

    # Convert dict to list with catalog_id included
    matches_list = [
        {"catalog_id": cid, **match_info} for cid, match_info in matches_dict.items()
    ]

    # Sort by metric (descending - higher is better for both metrics)
    sorted_matches = sorted(
        matches_list,
        key=lambda x: x.get(metric, 0.0),
        reverse=True,
    )

    # Add rank
    for rank, match in enumerate(sorted_matches, start=1):
        match["rank"] = rank

    # Return top_k if specified
    if top_k is not None:
        return sorted_matches[:top_k]

    return sorted_matches


def compute_match_metrics(matches: dict[str, torch.Tensor]) -> dict[str, float]:
    """Compute all matching metrics for a single match result.

    Args:
        matches: Dictionary from LightGlue matcher

    Returns:
        Dictionary with computed metrics:
            - wasserstein: float
            - auc: float
            - num_matches: int

    Raises:
        KeyError: If matches dict is missing required keys
    """
    try:
        scores = extract_match_scores(matches)

        return {
            "wasserstein": compute_wasserstein_distance(scores),
            "auc": compute_auc(scores),
            "num_matches": len(scores),
        }
    except KeyError as e:
        raise KeyError(f"Failed to compute metrics: {e}")
