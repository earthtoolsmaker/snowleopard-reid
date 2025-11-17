"""Image processing utilities for the Snow Leopard Re-ID project."""

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def resize_image_if_needed_pil(image: Image.Image, max_dim: int = 1024) -> Image.Image:
    """
    Resize PIL Image if either dimension exceeds max_dim, maintaining aspect ratio.

    Args:
        image: PIL Image to resize
        max_dim: Maximum allowed dimension (default: 1024)

    Returns:
        Resized image (or original if no resize needed)
    """
    width, height = image.size

    if width <= max_dim and height <= max_dim:
        return image

    # Calculate scaling factor
    scale = min(max_dim / width, max_dim / height)

    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize image using high-quality LANCZOS filter
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")

    return resized


def resize_image_if_needed_cv2(img: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """
    Resize cv2 image if either dimension exceeds max_dim, maintaining aspect ratio.

    Args:
        img: Input image as numpy array
        max_dim: Maximum allowed dimension (default: 1024)

    Returns:
        Resized image (or original if no resize needed)
    """
    height, width = img.shape[:2]

    if height <= max_dim and width <= max_dim:
        return img

    # Calculate scaling factor
    scale = min(max_dim / height, max_dim / width)

    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize image using INTER_AREA (best for downscaling)
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")

    return resized


def resize_image_if_needed(
    image: Image.Image | np.ndarray, max_dim: int = 1024
) -> Image.Image | np.ndarray:
    """
    Resize image if either dimension exceeds max_dim, maintaining aspect ratio.

    Automatically detects whether the input is a PIL Image or numpy array
    and applies the appropriate resize function.

    Args:
        image: PIL Image or numpy array to resize
        max_dim: Maximum allowed dimension (default: 1024)

    Returns:
        Resized image (or original if no resize needed)
    """
    if isinstance(image, Image.Image):
        return resize_image_if_needed_pil(image=image, max_dim=max_dim)
    elif isinstance(image, np.ndarray):
        return resize_image_if_needed_cv2(img=image, max_dim=max_dim)
    else:
        raise TypeError(f"Expected PIL.Image.Image or np.ndarray, got {type(image)}")
