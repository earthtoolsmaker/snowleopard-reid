"""Utilities for feature extraction and management.

This module provides functions for extracting, loading, and saving image features
using various feature extractors (SIFT, SuperPoint, DISK, ALIKED).
"""

from pathlib import Path

import torch
from lightglue import SIFT
from lightglue.utils import load_image, rbd
from PIL import Image


def extract_sift_features(
    image_path: Path | str | Image.Image,
    max_num_keypoints: int = 2048,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Extract SIFT features from an image.

    Args:
        image_path: Path to image file or PIL Image object
        max_num_keypoints: Maximum number of keypoints to extract (default: 2048, range: 512-4096)
        device: Device to run extraction on ('cpu' or 'cuda')

    Returns:
        Dictionary with keys:
            - keypoints: Tensor of shape [N, 2] with (x, y) coordinates
            - descriptors: Tensor of shape [N, 128] with SIFT descriptors
            - scores: Tensor of shape [N] with keypoint scores
            - image_size: Tensor of shape [2] with (width, height)

    Raises:
        ValueError: If max_num_keypoints is out of valid range
        FileNotFoundError: If image_path is a string/Path and file doesn't exist
    """
    # Validate parameters
    if not (512 <= max_num_keypoints <= 4096):
        raise ValueError(
            f"max_num_keypoints must be in range 512-4096, got {max_num_keypoints}"
        )

    # Initialize extractor
    extractor = SIFT(max_num_keypoints=max_num_keypoints).eval().to(device)

    # Load image
    if isinstance(image_path, (str, Path)):
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        # load_image returns torch.Tensor [3, H, W]
        image = load_image(str(image_path))
    elif isinstance(image_path, Image.Image):
        # Convert PIL Image to path temporarily
        # For now, require path input for lightglue compatibility
        raise TypeError(
            "PIL Image input not yet supported, please provide path to image file"
        )
    else:
        raise TypeError(
            f"image_path must be str, Path, or PIL Image, got {type(image_path)}"
        )

    # Move image to device
    image = image.to(device)

    # Extract features
    with torch.no_grad():
        feats = extractor.extract(image)  # auto-resizes image
        feats = rbd(feats)  # remove batch dimension

    # Move features back to CPU for storage
    if device != "cpu":
        feats = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in feats.items()
        }

    return feats


def load_features(features_path: Path | str) -> dict[str, torch.Tensor]:
    """Load features from a PyTorch .pt file.

    Args:
        features_path: Path to .pt file containing features

    Returns:
        Dictionary with feature tensors (keypoints, descriptors, scores, etc.)

    Raises:
        FileNotFoundError: If features file doesn't exist
        RuntimeError: If file cannot be loaded
    """
    features_path = Path(features_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    try:
        feats = torch.load(features_path, map_location="cpu", weights_only=False)
        return feats
    except Exception as e:
        raise RuntimeError(f"Failed to load features from {features_path}: {e}")


def save_features(
    features: dict[str, torch.Tensor],
    output_path: Path | str,
    create_dirs: bool = True,
) -> None:
    """Save features to a PyTorch .pt file.

    Args:
        features: Dictionary with feature tensors to save
        output_path: Path where to save .pt file
        create_dirs: Whether to create parent directories if they don't exist

    Raises:
        ValueError: If features dict is empty or invalid
        OSError: If directory creation or file writing fails
    """
    if not features:
        raise ValueError("Features dictionary is empty")

    output_path = Path(output_path)

    # Create parent directories if needed
    if create_dirs and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.save(features, output_path)
    except Exception as e:
        raise OSError(f"Failed to save features to {output_path}: {e}")


def get_num_keypoints(features: dict[str, torch.Tensor]) -> int:
    """Get the number of keypoints from a features dictionary.

    Args:
        features: Dictionary with 'keypoints' tensor

    Returns:
        Number of keypoints (first dimension of keypoints tensor)

    Raises:
        KeyError: If 'keypoints' key is missing
    """
    if "keypoints" not in features:
        raise KeyError("Features dictionary missing 'keypoints' key")

    return features["keypoints"].shape[0]


def extract_features(
    extractor: str,
    image_path: Path | str | Image.Image,
    max_num_keypoints: int = 2048,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Extract features from an image using the specified extractor.

    Factory function that dispatches to the appropriate feature extractor.

    Args:
        extractor: Feature extractor name ('sift', 'superpoint', 'disk', 'aliked')
        image_path: Path to image file or PIL Image object
        max_num_keypoints: Maximum number of keypoints to extract (default: 2048)
        device: Device to run extraction on ('cpu' or 'cuda')

    Returns:
        Dictionary with feature tensors (keypoints, descriptors, scores, image_size)

    Raises:
        ValueError: If extractor name is not supported
        FileNotFoundError: If image_path is a string/Path and file doesn't exist

    Examples:
        >>> features = extract_features("sift", "image.jpg")
        >>> features = extract_features("sift", "image.jpg", max_num_keypoints=4096, device="cuda")
    """
    extractor = extractor.lower()

    if extractor == "sift":
        return extract_sift_features(image_path, max_num_keypoints, device)
    else:
        raise ValueError(
            f"Unsupported extractor: {extractor}. Supported extractors: sift"
        )
