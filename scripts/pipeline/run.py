"""Run snow leopard matching pipeline to identify individual leopards from query images.

This script orchestrates the complete end-to-end snow leopard identification pipeline, combining
deep learning segmentation (YOLO), traditional feature extraction (SIFT), and modern
feature matching (LightGlue) to identify leopards against a reference catalog.

This script:
1. YOLO segmentation to detect all leopards in image
2. Mask selection to choose best leopard mask (confidence_area strategy)
3. Preprocessing to crop and mask the selected leopard
4. Feature extraction using SIFT (2048 keypoints)
5. Matching against catalog using LightGlue and Wasserstein distance

The pipeline exports artifacts at each stage for inspection and debugging, enabling
detailed analysis of where identification succeeds or fails.

Usage:
    python scripts/pipeline/run.py --image-path <path> --output-dir <path> [options]

Example:
    python scripts/pipeline/run.py \\
        --image-path ./data/test/query_leopard.jpg \\
        --catalog-path ./data/08_catalog/v1.0/ \\
        --model-path ./data/04_models/yolo/segmentation/best/weights/best.pt \\
        --output-dir ./data/06_matching_results/query_001/ \\
        --top-k 5

Technical Details:
    - YOLO model detects multiple leopards; best mask selected via confidence * area
    - SIFT extracts up to 2048 keypoints from cropped/masked leopard image
    - LightGlue performs robust feature matching between query and catalog images
    - Wasserstein distance measures similarity of match distributions
    - Lower Wasserstein distance indicates better match quality
    - All intermediate outputs saved to output_dir/pipeline/stages/ for debugging
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO

from snowleopard_reid.catalog import load_catalog_index
from snowleopard_reid.features import save_features
from snowleopard_reid.pipeline.stages import (
    run_feature_extraction_stage,
    run_mask_selection_stage,
    run_matching_stage,
    run_preprocess_stage,
    run_segmentation_stage,
)


def setup_logging(log_level: str) -> None:
    """Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level = getattr(logging, log_level.upper())
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress noisy third-party loggers unless in DEBUG mode
    if level > logging.DEBUG:
        logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.WARNING)


def save_stage_artifacts(output_dir: Path, stage: dict, stage_number: int) -> None:
    """Save artifacts for a pipeline stage."""
    logger = logging.getLogger(__name__)

    # Create numbered directory: pipeline/stages/{number:02d}_{stage_id}/
    stage_id = stage["stage_id"]
    stage_dir = output_dir / "pipeline" / "stages" / f"{stage_number:02d}_{stage_id}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    if stage_id == "segmentation":
        _save_yolo_artifacts(stage_dir=stage_dir, data=stage["data"])
    elif stage_id == "mask_selection":
        _save_selection_artifacts(stage_dir=stage_dir, data=stage["data"])
    elif stage_id == "preprocessing":
        _save_preprocessing_artifacts(stage_dir=stage_dir, data=stage["data"])
    elif stage_id == "feature_extraction":
        _save_features_artifacts(stage_dir=stage_dir, data=stage["data"])
    elif stage_id == "matching":
        _save_matching_artifacts(stage_dir=stage_dir, data=stage["data"])

    logger.info(f"Saved artifacts to {stage_dir}")


def _save_yolo_artifacts(stage_dir: Path, data: dict) -> None:
    """Save YOLO stage artifacts."""
    # Save predictions JSON (without masks - too large)
    predictions_json = {
        "image_path": data["image_path"],
        "image_size": data["image_size"],
        "num_predictions": len(data["predictions"]),
        "predictions_summary": [
            {
                "index": idx,
                "confidence": pred["confidence"],
                "bbox_xywhn": pred["bbox_xywhn"],
                "class_name": pred["class_name"],
            }
            for idx, pred in enumerate(data["predictions"])
        ],
    }

    with open(stage_dir / "predictions.json", "w") as f:
        json.dump(predictions_json, f, indent=2)

    # Save visualization if predictions exist
    if data["predictions"]:
        image = cv2.imread(data["image_path"])
        image_height, image_width = image.shape[:2]

        # Create overlay for masks and bboxes
        overlay = image.copy()

        for idx, pred in enumerate(data["predictions"]):
            # Draw mask overlay (green with transparency)
            # Resize mask to match image dimensions (YOLO masks are at inference resolution)
            mask = pred["mask"]
            mask_resized = cv2.resize(
                mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST
            )

            colored_mask = np.zeros_like(image)
            colored_mask[mask_resized > 0] = [0, 255, 0]  # Green in BGR
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

            # Convert normalized xywh to pixel xyxy
            bbox = pred["bbox_xywhn"]
            x_center = bbox["x_center"] * image_width
            y_center = bbox["y_center"] * image_height
            w = bbox["width"] * image_width
            h = bbox["height"] * image_height

            x_min = int(x_center - w / 2)
            y_min = int(y_center - h / 2)
            x_max = int(x_center + w / 2)
            y_max = int(y_center + h / 2)

            # Draw bbox
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Draw label
            label = f"{pred['class_name']} {pred['confidence']:.2f}"
            cv2.putText(
                overlay,
                label,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imwrite(str(stage_dir / "visualization.jpg"), overlay)


def _save_selection_artifacts(stage_dir: Path, data: dict) -> None:
    """Save selection stage artifacts."""
    # Save selection metadata
    with open(stage_dir / "selected_mask.json", "w") as f:
        json.dump(data["metadata"], f, indent=2)

    # Save binary mask as PNG
    mask = data["selected_prediction"]["mask"]
    mask_image = Image.fromarray(mask)
    mask_image.save(stage_dir / "mask_binary.png")


def _save_preprocessing_artifacts(stage_dir: Path, data: dict) -> None:
    """Save preprocessing stage artifacts."""
    # Save cropped image
    cropped_image = data["cropped_image"]
    cropped_image.save(stage_dir / "cropped.jpg", quality=95)

    # Save metadata
    with open(stage_dir / "metadata.json", "w") as f:
        json.dump(data["metadata"], f, indent=2)


def _save_features_artifacts(stage_dir: Path, data: dict) -> None:
    """Save features stage artifacts."""
    # Save features as PyTorch tensor
    features = data["features"]
    save_features(features=features, output_path=stage_dir / "features.pt")

    # Save metadata
    metadata = {
        "num_keypoints": features["keypoints"].shape[0],
        "descriptor_dim": features["descriptors"].shape[1],
        "image_size": features["image_size"].tolist(),
    }
    with open(stage_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def _save_matching_artifacts(stage_dir: Path, data: dict) -> None:
    """Save matching stage artifacts."""
    # Save matches JSON
    with open(stage_dir / "matches.json", "w") as f:
        json.dump(data, f, indent=2)


def _create_pipeline_manifest(
    output_dir: Path,
    query_id: str,
    image_path: str,
    stages: list[dict],
) -> None:
    """Create pipeline manifest YAML with links to all artifacts.

    Args:
        output_dir: Root output directory
        query_id: Query identifier
        image_path: Path to query image
        stages: List of stage dictionaries with stage_number, stage_id, stage_name
    """
    # Define artifact mappings for each stage
    stage_artifacts = {
        "segmentation": {
            "predictions": "predictions.json",
            "visualization": "visualization.jpg",
        },
        "mask_selection": {
            "metadata": "selected_mask.json",
            "mask": "mask_binary.png",
        },
        "preprocessing": {
            "metadata": "metadata.json",
            "cropped_image": "cropped.jpg",
        },
        "feature_extraction": {
            "metadata": "metadata.json",
            "features": "features.pt",
        },
        "matching": {
            "matches": "matches.json",
            "pairwise_dir": "pairwise/",
        },
    }

    # Build manifest structure
    manifest = {
        "format_version": "1.0",
        "query": {
            "query_id": query_id,
            "image_path": image_path,
        },
        "predictions_file": "predictions.json",
        "pipeline": {
            "stages": [],
        },
    }

    # Add stage information
    for stage_number, stage in enumerate(stages, start=1):
        stage_id = stage["stage_id"]
        stage_name = stage["stage_name"]
        stage_dir = f"pipeline/stages/{stage_number:02d}_{stage_id}"

        # Build artifacts dict with full paths
        artifacts = {}
        for artifact_key, artifact_file in stage_artifacts[stage_id].items():
            artifacts[artifact_key] = f"{stage_dir}/{artifact_file}"

        manifest["pipeline"]["stages"].append(
            {
                "id": stage_id,
                "name": stage_name,
                "directory": stage_dir,
                "artifacts": artifacts,
            }
        )

    # Save manifest YAML
    manifest_path = output_dir / "pipeline" / "manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)


def run(
    image_path: Path,
    catalog_path: Path,
    model_path: Path,
    output_dir: Path,
    top_k: int = 5,
    extractor: str = "sift",
    max_keypoints: int = 2048,
    confidence_threshold: float = 0.5,
    mask_selection_strategy: str = "confidence_area",
    padding: int = 5,
    device: str | None = None,
    export_artifacts: bool = True,
) -> dict:
    """Run full matching pipeline.

    Args:
        image_path: Path to query image
        catalog_path: Path to catalog root directory
        model_path: Path to YOLO model checkpoint
        output_dir: Directory to save results
        top_k: Number of top matches to return
        extractor: Feature extractor to use
        max_keypoints: Maximum keypoints to extract
        confidence_threshold: YOLO confidence threshold
        mask_selection_strategy: Strategy for selecting best mask
        padding: Padding around mask bbox
        device: Device to use (None = auto-detect)
        export_artifacts: Whether to export intermediate artifacts

    Returns:
        Dictionary with complete pipeline output
    """
    logger = logging.getLogger(__name__)

    # Early validation: check if extractor is available in catalog
    logger.info("Validating extractor availability in catalog...")
    try:
        catalog_index = load_catalog_index(catalog_path)
        available_extractors = list(catalog_index.get("feature_extractors", {}).keys())

        if not available_extractors:
            raise ValueError(
                f"No feature extractors found in catalog at {catalog_path}. "
                f"Please extract features first using: "
                f"uv run python scripts/catalog/extract_catalog_features.py "
                f"--catalog-dir {catalog_path} --extractor {extractor}"
            )

        if extractor not in available_extractors:
            raise ValueError(
                f"Extractor '{extractor}' not available in catalog. "
                f"Available extractors: {', '.join(available_extractors)}. "
                f"Please extract features first using: "
                f"uv run python scripts/catalog/extract_catalog_features.py "
                f"--catalog-dir {catalog_path} --extractor {extractor}"
            )

        logger.info(f"✓ Extractor '{extractor}' is available in catalog")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Catalog not found at {catalog_path}. "
            f"Please ensure the catalog exists and has been built."
        )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Stage 1: YOLO Segmentation
    logger.info("=" * 60)
    logger.info("STAGE 1: YOLO Segmentation")
    logger.info("=" * 60)
    logger.info(f"Loading YOLO model from {model_path}")
    model = YOLO(str(model_path))
    stage1 = run_segmentation_stage(
        model=model,
        image_path=image_path,
        confidence_threshold=confidence_threshold,
        device=device,
    )
    if export_artifacts:
        save_stage_artifacts(output_dir=output_dir, stage=stage1, stage_number=1)

    # Stage 2: Mask Selection
    logger.info("=" * 60)
    logger.info("STAGE 2: Mask Selection")
    logger.info("=" * 60)
    predictions = stage1["data"]["predictions"]
    if not predictions:
        raise RuntimeError("No predictions found in YOLO stage")
    stage2 = run_mask_selection_stage(
        predictions=predictions,
        strategy=mask_selection_strategy,
    )
    if export_artifacts:
        save_stage_artifacts(output_dir=output_dir, stage=stage2, stage_number=2)

    # Stage 3: Preprocessing
    logger.info("=" * 60)
    logger.info("STAGE 3: Preprocessing")
    logger.info("=" * 60)
    selected_prediction = stage2["data"]["selected_prediction"]
    mask = selected_prediction["mask"]
    stage3 = run_preprocess_stage(
        image_path=image_path,
        mask=mask,
        padding=padding,
    )
    if export_artifacts:
        save_stage_artifacts(output_dir=output_dir, stage=stage3, stage_number=3)

    # Stage 4: Feature Extraction
    logger.info("=" * 60)
    logger.info("STAGE 4: Feature Extraction")
    logger.info("=" * 60)
    cropped_image = stage3["data"]["cropped_image"]
    stage4 = run_feature_extraction_stage(
        image=cropped_image,
        extractor=extractor,
        max_keypoints=max_keypoints,
        device=device,
    )
    if export_artifacts:
        save_stage_artifacts(output_dir=output_dir, stage=stage4, stage_number=4)

    # Stage 5: Matching
    logger.info("=" * 60)
    logger.info("STAGE 5: Matching")
    logger.info("=" * 60)
    query_features = stage4["data"]["features"]
    pairwise_output_dir = None
    if export_artifacts:
        pairwise_output_dir = (
            output_dir / "pipeline" / "stages" / "05_matching" / "pairwise"
        )
    stage5 = run_matching_stage(
        query_features=query_features,
        catalog_path=catalog_path,
        top_k=top_k,
        extractor=extractor,
        device=device,
        query_image_path=str(image_path),
        pairwise_output_dir=pairwise_output_dir,
    )
    if export_artifacts:
        save_stage_artifacts(output_dir=output_dir, stage=stage5, stage_number=5)

    # Create pipeline metadata if exporting artifacts
    if export_artifacts:
        _create_pipeline_manifest(
            output_dir=output_dir,
            query_id=output_dir.name,
            image_path=str(image_path),
            stages=[stage1, stage2, stage3, stage4, stage5],
        )

    # Build complete pipeline output
    pipeline_output = {
        "format_version": "1.0",
        "query": {
            "image_path": str(image_path),
            "query_id": output_dir.name,
        },
        "config": {
            "catalog_path": str(catalog_path),
            "model_path": str(model_path),
            "top_k": top_k,
            "extractor": extractor,
            "max_keypoints": max_keypoints,
            "confidence_threshold": confidence_threshold,
            "mask_selection_strategy": mask_selection_strategy,
            "padding": padding,
            "device": device or "auto",
        },
        "results": stage5["data"],
        "output_dir": str(output_dir),
    }

    # Save predictions output
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(pipeline_output, f, indent=2)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Top match: {stage5['data']['matches'][0]['leopard_name']}")
    logger.info(f"Wasserstein: {stage5['data']['matches'][0]['wasserstein']:.4f}")
    logger.info(f"Num matches: {stage5['data']['matches'][0]['num_matches']}")
    logger.info(f"Results saved to: {output_dir}")

    return pipeline_output


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run snow leopard matching pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/pipeline/run.py \\
    --image-path data/test/query.jpg \\
    --output-dir data/06_matching_results/query_001/

  # Custom model and catalog
  python scripts/pipeline/run.py \\
    --image-path data/test/query.jpg \\
    --catalog-path data/08_catalog/v1.0/ \\
    --model-path data/04_models/yolo/segmentation/baseline/weights/best.pt \\
    --output-dir data/06_matching_results/query_001/

  # Top-10 matches with baseline model
  python scripts/pipeline/run.py \\
    --image-path data/test/query.jpg \\
    --model-path data/04_models/yolo/segmentation/baseline/weights/best.pt \\
    --top-k 10 \\
    --output-dir data/06_matching_results/query_001/

  # Debug mode with verbose logging
  python scripts/pipeline/run.py \\
    --image-path data/test/query.jpg \\
    --output-dir data/06_matching_results/query_001/ \\
    --log-level DEBUG
        """,
    )

    parser.add_argument(
        "--image-path",
        type=Path,
        required=True,
        help="Path to query image",
    )
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=Path("data/08_catalog/v1.0/"),
        help="Path to catalog root directory (default: data/08_catalog/v1.0/)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/04_models/yolo/segmentation/best/weights/best.pt"),
        help="Path to YOLO model (default: best model)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top matches to return (default: 5)",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default="sift",
        choices=["sift"],
        help="Feature extractor (default: sift)",
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=2048,
        help="Maximum keypoints to extract (default: 2048)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="YOLO confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--mask-selection-strategy",
        type=str,
        default="confidence_area",
        choices=["confidence_area", "confidence", "area", "center"],
        help="Mask selection strategy (default: confidence_area)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=5,
        help="Padding around mask bbox in pixels (default: 5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--no-export-artifacts",
        action="store_true",
        help="Disable exporting intermediate artifacts",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        run(
            image_path=args.image_path,
            catalog_path=args.catalog_path,
            model_path=args.model_path,
            output_dir=args.output_dir,
            top_k=args.top_k,
            extractor=args.extractor,
            max_keypoints=args.max_keypoints,
            confidence_threshold=args.confidence_threshold,
            mask_selection_strategy=args.mask_selection_strategy,
            padding=args.padding,
            device=args.device,
            export_artifacts=not args.no_export_artifacts,
        )
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
