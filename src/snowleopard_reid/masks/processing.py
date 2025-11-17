"""Utilities for mask processing and image cropping.

This module provides functions for working with binary segmentation masks,
calculating bounding boxes, and cropping images with masks applied.
"""

import numpy as np
from PIL import Image


def get_mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Calculate the tight bounding box of a binary mask.

    Args:
        mask: Binary mask array (0=background, 255=foreground)

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in pixel coordinates

    Raises:
        ValueError: If mask is empty (no foreground pixels)
    """
    # Find all pixels that are part of the mask
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        raise ValueError("Mask is empty (no foreground pixels)")

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return int(x_min), int(y_min), int(x_max), int(y_max)


def add_padding_to_bbox(
    bbox: tuple[int, int, int, int],
    padding: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Add padding to a bounding box, clamped to image boundaries.

    Args:
        bbox: Original bounding box (x_min, y_min, x_max, y_max)
        padding: Padding in pixels to add on all sides
        image_width: Image width for clamping
        image_height: Image height for clamping

    Returns:
        Padded bounding box (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, x_max, y_max = bbox

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image_width - 1, x_max + padding)
    y_max = min(image_height - 1, y_max + padding)

    return x_min, y_min, x_max, y_max


def crop_and_mask_image(
    image: Image.Image,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> Image.Image:
    """Crop image to bbox and apply mask with black background.

    Args:
        image: Original PIL Image
        mask: Binary mask array (same size as image, 0=background, 255=foreground)
        bbox: Bounding box to crop to (x_min, y_min, x_max, y_max)

    Returns:
        Cropped and masked PIL Image with black background
    """
    x_min, y_min, x_max, y_max = bbox

    # Crop image and mask to bounding box
    cropped_image = image.crop((x_min, y_min, x_max + 1, y_max + 1))
    cropped_mask = mask[y_min : y_max + 1, x_min : x_max + 1]

    # Convert image to numpy array
    image_array = np.array(cropped_image)

    # Create mask with correct shape (add channel dimension if needed)
    if len(image_array.shape) == 3:
        # RGB image - expand mask to 3 channels
        mask_3d = np.repeat(cropped_mask[:, :, np.newaxis] > 0, 3, axis=2)
    else:
        # Grayscale image
        mask_3d = cropped_mask > 0

    # Apply mask: keep original pixels where mask is True, black elsewhere
    masked_array = np.where(mask_3d, image_array, 0)

    # Convert back to PIL Image
    return Image.fromarray(masked_array.astype(np.uint8))
