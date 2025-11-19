"""Segmentation stage using YOLO or GDINO+SAM for snow leopard detection.

This module provides segmentation stages that detect and segment snow leopards
in query images using either:
1. YOLO (end-to-end learned segmentation)
2. GDINO+SAM (zero-shot detection + prompted segmentation)
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything_hq import SamPredictor, sam_model_registry
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import YOLO

from snowleopard_reid import get_device

logger = logging.getLogger(__name__)


def load_gdino_model(
    model_id: str = "IDEA-Research/grounding-dino-base",
    device: str | None = None,
) -> tuple[Any, Any]:
    """Load Grounding DINO model and processor.

    Args:
        model_id: HuggingFace model identifier
        device: Device to load model on (None = auto-detect)

    Returns:
        Tuple of (processor, model)
    """
    device = get_device(device=device, verbose=True)

    logger.info(f"Loading Grounding DINO model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model = model.to(device)
    model.eval()

    logger.info("Grounding DINO model loaded successfully")
    return processor, model


def load_sam_predictor(
    checkpoint_path: Path | str,
    model_type: str = "vit_l",
    device: str | None = None,
) -> SamPredictor:
    """Load SAM HQ predictor.

    Args:
        checkpoint_path: Path to SAM HQ checkpoint file
        model_type: Model type (vit_b, vit_l, vit_h)
        device: Device to load model on (None = auto-detect)

    Returns:
        SamPredictor instance
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")

    device_str = get_device(device=device, verbose=True)

    logger.info(f"Loading SAM HQ model: {model_type}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device_str)

    predictor = SamPredictor(sam)
    logger.info("SAM HQ model loaded successfully")

    return predictor


def _run_yolo_segmentation(
    model: YOLO,
    image_path: Path,
    confidence_threshold: float,
    device: str,
) -> dict:
    """Run YOLO segmentation (internal implementation).

    Args:
        model: Pre-loaded YOLO model
        image_path: Path to input image
        confidence_threshold: Minimum confidence to keep predictions
        device: Device to run on

    Returns:
        Standardized stage dict
    """
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
            "strategy": "yolo",
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


def _run_gdino_sam_segmentation(
    gdino_processor: Any,
    gdino_model: Any,
    sam_predictor: SamPredictor,
    image_path: Path,
    confidence_threshold: float,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> dict:
    """Run GDINO+SAM segmentation (internal implementation).

    Args:
        gdino_processor: Grounding DINO processor
        gdino_model: Grounding DINO model
        sam_predictor: SAM HQ predictor
        image_path: Path to input image
        confidence_threshold: Minimum confidence to keep predictions
        text_prompt: Text prompt for GDINO
        box_threshold: GDINO box threshold
        text_threshold: GDINO text threshold
        device: Device to run on

    Returns:
        Standardized stage dict
    """
    # Load image (PIL for GDINO, numpy for SAM)
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    image_height, image_width = image_np.shape[:2]

    # Run Grounding DINO detection
    logger.info("Running Grounding DINO detection...")
    inputs = gdino_processor(images=image_pil, text=text_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = gdino_model(**inputs)

    # Post-process GDINO outputs
    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image_pil.size[::-1]],  # (height, width)
    )[0]

    # Filter by confidence threshold
    labels = results.get("text_labels", results.get("labels", []))
    boxes = results["boxes"]
    scores = results["scores"]

    logger.info(f"GDINO detected {len(boxes)} objects")

    # Filter predictions by confidence threshold
    filtered_detections = [
        (box, score, label)
        for box, score, label in zip(boxes, scores, labels)
        if float(score) >= confidence_threshold
    ]

    logger.info(
        f"Filtered to {len(filtered_detections)} detections (confidence >= {confidence_threshold})"
    )

    if not filtered_detections:
        logger.warning(f"No detections found for {image_path}")
        predictions = []
    else:
        # Set image for SAM (do this once)
        logger.info("Running SAM HQ segmentation...")
        sam_predictor.set_image(image_np)

        predictions = []
        for idx, (box, gdino_score, label) in enumerate(filtered_detections):
            # Convert box to pixel coordinates and format for SAM
            x_min, y_min, x_max, y_max = box
            bbox_xyxy = np.array(
                [float(x_min), float(y_min), float(x_max), float(y_max)]
            )

            # Run SAM with bounding box prompt
            masks, sam_scores, logits = sam_predictor.predict(
                box=bbox_xyxy[None, :],
                multimask_output=False,
                hq_token_only=True,
            )

            # Get mask (first and only mask, since multimask_output=False)
            mask = masks[0]  # Shape: (H, W), boolean
            sam_score = float(sam_scores[0])

            # Convert mask to uint8 (0-255)
            mask_uint8 = (mask * 255).astype(np.uint8)

            # Convert bbox to normalized xywh format (same as YOLO)
            x_center = (float(x_min) + float(x_max)) / 2 / image_width
            y_center = (float(y_min) + float(y_max)) / 2 / image_height
            width = (float(x_max) - float(x_min)) / image_width
            height = (float(y_max) - float(y_min)) / image_height

            predictions.append(
                {
                    "mask": mask_uint8,
                    "confidence": float(
                        gdino_score
                    ),  # Use GDINO score as primary confidence
                    "bbox_xywhn": {
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                    },
                    "class_id": 0,  # Single class (snow leopard)
                    "class_name": label,
                    # Additional metadata
                    "sam_score": sam_score,
                    "gdino_score": float(gdino_score),
                }
            )

        logger.info(f"Generated {len(predictions)} segmentation masks")

    # Return standardized stage dict
    return {
        "stage_id": "segmentation",
        "stage_name": "GDINO+SAM Segmentation",
        "description": "Snow leopard detection using Grounding DINO and segmentation using SAM HQ",
        "config": {
            "strategy": "gdino_sam",
            "confidence_threshold": confidence_threshold,
            "text_prompt": text_prompt,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
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


def run_segmentation_stage(
    image_path: Path | str,
    strategy: str = "yolo",
    confidence_threshold: float = 0.5,
    device: str | None = None,
    # YOLO-specific parameters
    yolo_model: YOLO | None = None,
    # GDINO+SAM-specific parameters
    gdino_processor: Any | None = None,
    gdino_model: Any | None = None,
    sam_predictor: SamPredictor | None = None,
    text_prompt: str = "a snow leopard.",
    box_threshold: float = 0.30,
    text_threshold: float = 0.20,
) -> dict:
    """Run segmentation on query image using specified strategy.

    This stage performs snow leopard detection and segmentation using either:
    - YOLO: End-to-end learned segmentation
    - GDINO+SAM: Zero-shot detection + prompted segmentation

    Args:
        image_path: Path to input image
        strategy: Segmentation strategy ("yolo" or "gdino_sam")
        confidence_threshold: Minimum confidence to keep predictions (default: 0.5)
        device: Device to run on ('cpu', 'cuda', or None for auto-detect)
        yolo_model: Pre-loaded YOLO model (required if strategy="yolo")
        gdino_processor: Pre-loaded GDINO processor (required if strategy="gdino_sam")
        gdino_model: Pre-loaded GDINO model (required if strategy="gdino_sam")
        sam_predictor: Pre-loaded SAM predictor (required if strategy="gdino_sam")
        text_prompt: Text prompt for GDINO (default: "a snow leopard.")
        box_threshold: GDINO box confidence threshold (default: 0.30)
        text_threshold: GDINO text matching threshold (default: 0.20)

    Returns:
        Stage dict with structure:
        {
            "stage_id": "segmentation",
            "stage_name": str,
            "description": str,
            "config": {
                "strategy": str,
                "confidence_threshold": float,
                "device": str,
                ...
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
                        "bbox_xywhn": {...},
                        "class_id": int,
                        "class_name": str,
                        # Optional (GDINO+SAM only)
                        "sam_score": float,
                        "gdino_score": float,
                    },
                    ...
                ]
            }
        }

    Raises:
        ValueError: If strategy is invalid or required models are missing
        FileNotFoundError: If image doesn't exist
        RuntimeError: If inference fails
    """
    image_path = Path(image_path)

    # Validate inputs
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if strategy not in ["yolo", "gdino_sam"]:
        raise ValueError(f"Invalid strategy: {strategy}. Must be 'yolo' or 'gdino_sam'")

    # Auto-detect device if not specified
    device = get_device(device=device, verbose=True)

    # Dispatch to appropriate implementation
    if strategy == "yolo":
        if yolo_model is None:
            raise ValueError("yolo_model is required when strategy='yolo'")
        return _run_yolo_segmentation(
            model=yolo_model,
            image_path=image_path,
            confidence_threshold=confidence_threshold,
            device=device,
        )

    elif strategy == "gdino_sam":
        if gdino_processor is None or gdino_model is None or sam_predictor is None:
            raise ValueError(
                "gdino_processor, gdino_model, and sam_predictor are required when strategy='gdino_sam'"
            )
        return _run_gdino_sam_segmentation(
            gdino_processor=gdino_processor,
            gdino_model=gdino_model,
            sam_predictor=sam_predictor,
            image_path=image_path,
            confidence_threshold=confidence_threshold,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )
