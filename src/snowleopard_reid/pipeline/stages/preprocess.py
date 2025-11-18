"""Preprocessing stage for cropping and masking snow leopard images.

This module provides preprocessing operations to extract and mask the leopard region
from the full image.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def run_preprocess_stage(
    image_path: Path | str,
    mask: np.ndarray,
    padding: int = 5,
) -> dict:
    """Run preprocessing stage.

    This stage crops the image to the mask bounding box with padding and applies
    the mask to isolate the leopard region.

    Args:
        image_path: Path to input image
        mask: Binary mask (H×W, uint8) from segmentation
        padding: Padding around bbox in pixels (default: 5)

    Returns:
        Stage dict with structure:
        {
            "stage_id": "preprocessing",
            "stage_name": "Preprocessing",
            "description": "Crop and mask leopard region",
            "config": {
                "padding": int
            },
            "metrics": {
                "original_size": {"width": int, "height": int},
                "crop_size": {"width": int, "height": int}
            },
            "data": {
                "cropped_image": PIL.Image,
                "metadata": {
                    "original_size": {"width": int, "height": int},
                    "crop_bbox": {"x_min": int, "y_min": int, "x_max": int, "y_max": int},
                    "crop_size": {"width": int, "height": int}
                }
            }
        }

    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If mask is invalid
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    logger.info(f"Preprocessing image: {image_path}")

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width = image_rgb.shape[:2]

    # Resize mask to match image dimensions if needed
    if mask.shape[:2] != (image_height, image_width):
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        mask_resized = mask

    # Find bounding box of mask
    rows = np.any(mask_resized > 0, axis=1)
    cols = np.any(mask_resized > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        raise ValueError("Mask is empty (no pixels > 0)")

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Add padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image_width - 1, x_max + padding)
    y_max = min(image_height - 1, y_max + padding)

    # Crop image and mask
    cropped_image = image_rgb[y_min : y_max + 1, x_min : x_max + 1]
    cropped_mask = mask_resized[y_min : y_max + 1, x_min : x_max + 1]

    # Apply mask (set non-masked pixels to black)
    masked_image = cropped_image.copy()
    masked_image[cropped_mask == 0] = 0

    # Convert to PIL Image
    cropped_pil = Image.fromarray(masked_image)

    crop_height, crop_width = masked_image.shape[:2]

    logger.info(
        f"Cropped from {image_width}x{image_height} to {crop_width}x{crop_height} "
        f"(padding={padding}px)"
    )

    # Return standardized stage dict
    return {
        "stage_id": "preprocessing",
        "stage_name": "Preprocessing",
        "description": "Crop and mask leopard region",
        "config": {
            "padding": padding,
        },
        "metrics": {
            "original_size": {"width": image_width, "height": image_height},
            "crop_size": {"width": crop_width, "height": crop_height},
        },
        "data": {
            "cropped_image": cropped_pil,
            "metadata": {
                "original_size": {"width": image_width, "height": image_height},
                "crop_bbox": {
                    "x_min": int(x_min),
                    "y_min": int(y_min),
                    "x_max": int(x_max),
                    "y_max": int(y_max),
                },
                "crop_size": {"width": crop_width, "height": crop_height},
            },
        },
    }
