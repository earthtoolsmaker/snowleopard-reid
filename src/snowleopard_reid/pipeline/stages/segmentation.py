"""Segmentation stage using YOLO for snow leopard detection.

This module provides the YOLO segmentation stage that detects and segments snow leopards
in query images.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from snowleopard_reid import get_device

logger = logging.getLogger(__name__)


def run_segmentation_stage(
    model: YOLO,
    image_path: Path | str,
    confidence_threshold: float = 0.5,
    device: str | None = None,
) -> dict:
    """Run YOLO segmentation on query image.

    This stage performs snow leopard detection and segmentation using a YOLO model,
    returning predictions with masks and bounding boxes.

    Args:
        model: Pre-loaded YOLO model (required, cannot be None)
        image_path: Path to input image
        confidence_threshold: Minimum confidence to keep predictions (default: 0.5)
        device: Device to run on ('cpu', 'cuda', or None for auto-detect)

    Returns:
        Stage dict with structure:
        {
            "stage_id": "segmentation",
            "stage_name": "YOLO Segmentation",
            "description": "Snow leopard detection and segmentation using YOLO",
            "config": {
                "confidence_threshold": float,
                "device": str
            },
            "metrics": {
                "num_predictions": int
            },
            "data": {
                "image_path": str,
                "image_size": {"width": int, "height": int},
                "predictions": [
                    {
                        "mask": np.ndarray (H×W, uint8),
                        "confidence": float,
                        "bbox_xywhn": {"x_center": float, "y_center": float, "width": float, "height": float},
                        "class_id": int,
                        "class_name": str
                    },
                    ...
                ]
            }
        }

    Raises:
        FileNotFoundError: If image doesn't exist
        RuntimeError: If YOLO inference fails
    """
    image_path = Path(image_path)

    # Validate inputs
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Auto-detect device if not specified
    device = get_device(device=device, verbose=True)

    # Load image to get size
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    image_height, image_width = image.shape[:2]

    # Run inference
    try:
        results = model(
            str(image_path),
            conf=confidence_threshold,
            device=device,
            verbose=False,
        )
    except Exception as e:
        raise RuntimeError(f"YOLO inference failed: {e}")

    # Parse results
    predictions = []
    result = results[0]  # Single image, so single result

    # Debug: Print result attributes
    logger.info(f"Result object type: {type(result)}")
    logger.info(f"Result has boxes: {result.boxes is not None}")
    logger.info(f"Result has masks: {result.masks is not None}")
    if result.boxes is not None:
        logger.info(f"Number of boxes: {len(result.boxes)}")
    if result.masks is not None:
        logger.info(f"Number of masks: {len(result.masks)}")

    # Check if any detections found
    if result.masks is None or len(result.masks) == 0:
        logger.warning(f"No detections found for {image_path}")
        logger.warning(
            f"Boxes present: {result.boxes is not None}, Masks present: {result.masks is not None}"
        )
    else:
        # Extract masks and metadata
        for idx in range(len(result.masks)):
            # Get mask (binary, H×W)
            mask = result.masks.data[idx].cpu().numpy()  # Shape: (H, W)
            mask = (mask * 255).astype(np.uint8)  # Convert to 0-255

            # Get bounding box (normalized xywh format)
            bbox = result.boxes.xywhn[idx].cpu().numpy()  # Shape: (4,)
            x_center, y_center, width, height = bbox

            # Get confidence
            confidence = float(result.boxes.conf[idx].cpu().numpy())

            # Get class info
            class_id = int(result.boxes.cls[idx].cpu().numpy())
            class_name = result.names[class_id]

            predictions.append(
                {
                    "mask": mask,
                    "confidence": confidence,
                    "bbox_xywhn": {
                        "x_center": float(x_center),
                        "y_center": float(y_center),
                        "width": float(width),
                        "height": float(height),
                    },
                    "class_id": class_id,
                    "class_name": class_name,
                }
            )

        logger.info(
            f"Found {len(predictions)} predictions (confidence >= {confidence_threshold})"
        )

    # Return standardized stage dict
    return {
        "stage_id": "segmentation",
        "stage_name": "YOLO Segmentation",
        "description": "Snow leopard detection and segmentation using YOLO",
        "config": {
            "confidence_threshold": confidence_threshold,
            "device": device,
        },
        "metrics": {
            "num_predictions": len(predictions),
        },
        "data": {
            "image_path": str(image_path),
            "image_size": {"width": image_width, "height": image_height},
            "predictions": predictions,
        },
    }
