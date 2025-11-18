"""Mask selection stage for choosing the best snow leopard mask.

This module provides logic for selecting the best mask from multiple YOLO predictions.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def select_best_mask(
    predictions: list[dict],
    strategy: str = "confidence_area",
) -> tuple[int, dict]:
    """Select the best mask from predictions using specified strategy.

    Args:
        predictions: List of prediction dicts from YOLO segmentation stage
        strategy: Selection strategy ('confidence_area', 'confidence', 'area', 'center')

    Returns:
        Tuple of (selected_index, selected_prediction)

    Raises:
        ValueError: If predictions list is empty or strategy is invalid
    """
    if not predictions:
        raise ValueError("Predictions list is empty")

    valid_strategies = ["confidence_area", "confidence", "area", "center"]
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Valid strategies: {valid_strategies}"
        )

    if strategy == "confidence_area":
        # Select mask with highest confidence * area product
        scores = []
        for pred in predictions:
            confidence = pred["confidence"]
            mask = pred["mask"]
            area = np.sum(mask > 0)
            scores.append(confidence * area)
        selected_idx = int(np.argmax(scores))

    elif strategy == "confidence":
        # Select mask with highest confidence
        confidences = [pred["confidence"] for pred in predictions]
        selected_idx = int(np.argmax(confidences))

    elif strategy == "area":
        # Select mask with largest area
        areas = [np.sum(pred["mask"] > 0) for pred in predictions]
        selected_idx = int(np.argmax(areas))

    elif strategy == "center":
        # Select mask closest to image center
        # This strategy requires image size, which we can get from bbox
        distances = []
        for pred in predictions:
            bbox = pred["bbox_xywhn"]
            # Center is already normalized to [0, 1]
            x_center = bbox["x_center"]
            y_center = bbox["y_center"]
            # Distance from image center (0.5, 0.5)
            dist = np.sqrt((x_center - 0.5) ** 2 + (y_center - 0.5) ** 2)
            distances.append(dist)
        selected_idx = int(np.argmin(distances))

    return selected_idx, predictions[selected_idx]


def run_mask_selection_stage(
    predictions: list[dict],
    strategy: str = "confidence_area",
) -> dict:
    """Run mask selection stage.

    This stage selects the best mask from multiple YOLO predictions using
    the specified selection strategy.

    Args:
        predictions: List of prediction dicts from segmentation stage
        strategy: Selection strategy (default: 'confidence_area')

    Returns:
        Stage dict with structure:
        {
            "stage_id": "mask_selection",
            "stage_name": "Mask Selection",
            "description": "Select best mask from predictions",
            "config": {
                "strategy": str
            },
            "metrics": {
                "num_candidates": int,
                "selected_index": int,
                "selected_confidence": float
            },
            "data": {
                "selected_prediction": dict,
                "metadata": {
                    "strategy": str,
                    "selected_index": int,
                    "num_candidates": int,
                    "confidence": float,
                    "mask_area": int
                }
            }
        }

    Raises:
        ValueError: If predictions list is empty
    """
    logger.info(f"Selecting best mask using strategy: {strategy}")

    # Select best mask
    selected_idx, selected_pred = select_best_mask(predictions, strategy)

    # Compute metadata
    mask_area = int(np.sum(selected_pred["mask"] > 0))
    confidence = selected_pred["confidence"]

    logger.info(
        f"Selected mask {selected_idx} (confidence={confidence:.3f}, area={mask_area})"
    )

    # Return standardized stage dict
    return {
        "stage_id": "mask_selection",
        "stage_name": "Mask Selection",
        "description": "Select best mask from predictions",
        "config": {
            "strategy": strategy,
        },
        "metrics": {
            "num_candidates": len(predictions),
            "selected_index": selected_idx,
            "selected_confidence": confidence,
        },
        "data": {
            "selected_prediction": selected_pred,
            "metadata": {
                "strategy": strategy,
                "selected_index": selected_idx,
                "num_candidates": len(predictions),
                "confidence": confidence,
                "mask_area": mask_area,
            },
        },
    }
