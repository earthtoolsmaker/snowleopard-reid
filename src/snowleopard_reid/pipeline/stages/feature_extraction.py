"""Feature extraction stage for query images.

This module extracts features from cropped query images for matching
against the catalog.
"""

import logging
import tempfile
from pathlib import Path

import torch
from PIL import Image

from snowleopard_reid import features, get_device

logger = logging.getLogger(__name__)


def run_feature_extraction_stage(
    image: Image.Image | Path | str,
    extractor: str = "sift",
    max_keypoints: int = 2048,
    device: str | None = None,
) -> dict:
    """Extract features from query image.

    This stage extracts keypoints and descriptors from the preprocessed query
    image using the specified feature extractor.

    Args:
        image: PIL Image object or path to image file
        extractor: Feature extractor to use (default: 'sift')
        max_keypoints: Maximum number of keypoints to extract (default: 2048)
        device: Device to run on ('cpu', 'cuda', or None for auto-detect)

    Returns:
        Stage dict with structure:
        {
            "stage_id": "feature_extraction",
            "stage_name": "Feature Extraction",
            "description": "Extract keypoints and descriptors",
            "config": {
                "extractor": str,
                "max_keypoints": int,
                "device": str
            },
            "metrics": {
                "num_keypoints": int
            },
            "data": {
                "features": {
                    "keypoints": torch.Tensor [N, 2],
                    "descriptors": torch.Tensor [N, D],
                    "scores": torch.Tensor [N],
                    "image_size": torch.Tensor [2]
                }
            }
        }

    Raises:
        ValueError: If extractor is not supported
        FileNotFoundError: If image path doesn't exist
        RuntimeError: If feature extraction fails
    """
    # Auto-detect device if not specified
    device = get_device(device=device, verbose=True)

    # Extract features using the factory function
    features_dict = _extract_features_from_image(
        image=image, extractor=extractor, max_keypoints=max_keypoints, device=device
    )

    # Get number of keypoints
    num_kpts = features.get_num_keypoints(features_dict)
    logger.info(f"Extracted {num_kpts} keypoints using {extractor.upper()}")

    # Return standardized stage dict
    return {
        "stage_id": "feature_extraction",
        "stage_name": "Feature Extraction",
        "description": "Extract keypoints and descriptors",
        "config": {
            "extractor": extractor,
            "max_keypoints": max_keypoints,
            "device": device,
        },
        "metrics": {
            "num_keypoints": num_kpts,
        },
        "data": {
            "features": features_dict,
        },
    }


def _extract_features_from_image(
    image: Image.Image | Path | str,
    extractor: str,
    max_keypoints: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Extract features from PIL Image or path using specified extractor.

    This is a wrapper that handles PIL Image input by saving to a temporary file,
    since lightglue's load_image() requires a file path.

    Args:
        image: PIL Image or path to image
        extractor: Feature extractor to use ('sift', 'superpoint', 'disk', 'aliked')
        max_keypoints: Maximum keypoints to extract
        device: Device to use

    Returns:
        Features dictionary
    """
    if isinstance(image, Image.Image):
        # Save PIL Image to temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            image.save(tmp_path, quality=95)

        try:
            # Extract features from temporary file
            feats = features.extract_features(
                extractor, tmp_path, max_keypoints, device
            )
        finally:
            # Clean up temporary file
            tmp_path.unlink()

        return feats
    else:
        # Image is already a path
        return features.extract_features(extractor, image, max_keypoints, device)
