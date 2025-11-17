"""Extract SIFT features from catalog images for snow leopard identification.

This script processes all images in a catalog database to extract SIFT keypoint features
that will be used for matching query leopards against the reference catalog.

This script:
1. Scans catalog database for all individual directories
2. Finds all images for each individual (organized by location/body_part)
3. Extracts SIFT features (default: 2048 max keypoints)
4. Saves features as .pt files in parallel directory structure
5. Logs progress and statistics

Usage:
    python scripts/catalog/extract_catalog_features.py --catalog-dir <path> [options]

Example:
    python scripts/catalog/extract_catalog_features.py \
        --catalog-dir ./data/08_catalog/v1.0/database \
        --extractor sift \
        --max-keypoints 2048
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def setup_logging(verbose: bool) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def extract_sift_features(
    image_path: Path,
    max_num_keypoints: int = 2048,
) -> dict:
    """Extract SIFT features from an image.

    Args:
        image_path: Path to input image
        max_num_keypoints: Maximum number of keypoints to extract

    Returns:
        Dictionary containing keypoints, descriptors, scores, and image_size as PyTorch tensors
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create(nfeatures=max_num_keypoints)

    # Detect and compute keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) == 0:
        # Return empty features if no keypoints found
        return {
            "keypoints": torch.empty((0, 2), dtype=torch.float32),
            "descriptors": torch.empty((0, 128), dtype=torch.float32),
            "scores": torch.empty((0,), dtype=torch.float32),
            "image_size": torch.tensor([img.shape[1], img.shape[0]], dtype=torch.int32),
        }

    # Convert keypoints to numpy arrays
    kpts = np.array([kp.pt for kp in keypoints], dtype=np.float32)  # (x, y) coordinates
    scores = np.array([kp.response for kp in keypoints], dtype=np.float32)

    # Limit to max_num_keypoints (already done by SIFT_create but be safe)
    if len(kpts) > max_num_keypoints:
        # Sort by response score and take top-k
        idx = np.argsort(scores)[::-1][:max_num_keypoints]
        kpts = kpts[idx]
        descriptors = descriptors[idx]
        scores = scores[idx]

    # Convert to PyTorch tensors
    features = {
        "keypoints": torch.from_numpy(kpts),  # Shape: [N, 2]
        "descriptors": torch.from_numpy(descriptors),  # Shape: [N, 128]
        "scores": torch.from_numpy(scores),  # Shape: [N]
        "image_size": torch.tensor([img.shape[1], img.shape[0]], dtype=torch.int32),
    }

    return features


def extract_catalog_features(
    catalog_dir: Path,
    extractor_name: str = "sift",
    max_num_keypoints: int = 2048,
    verbose: bool = False,
) -> None:
    """Extract features for all images in catalog.

    Args:
        catalog_dir: Path to catalog database directory
        extractor_name: Feature extractor to use (currently only 'sift' supported)
        max_num_keypoints: Maximum number of keypoints to extract per image
        verbose: Enable verbose logging
    """
    setup_logging(verbose)

    # Validate parameters
    if extractor_name.lower() != "sift":
        raise ValueError(f"Currently only 'sift' extractor is supported, got: {extractor_name}")

    if not (512 <= max_num_keypoints <= 4096):
        raise ValueError(
            f"max_num_keypoints must be in range 512-4096, got {max_num_keypoints}"
        )

    if not catalog_dir.exists():
        raise FileNotFoundError(f"Catalog directory not found: {catalog_dir}")

    logging.info(f"Initializing SIFT feature extractor (max_keypoints: {max_num_keypoints})")

    # Find all individual directories
    individual_dirs = [d for d in catalog_dir.iterdir() if d.is_dir()]
    logging.info(f"Found {len(individual_dirs)} individual directories")

    # Process each individual
    total_images = 0
    total_keypoints = 0

    for individual_dir in sorted(individual_dirs):
        individual_name = individual_dir.name
        images_dir = individual_dir / "images"
        features_dir = individual_dir / "features" / extractor_name

        if not images_dir.exists():
            logging.warning(f"No images directory for {individual_name}, skipping")
            continue

        # Find all images (recursively to handle location/body_part structure)
        image_files = list(images_dir.rglob("*.jpg"))
        logging.info(f"Processing {individual_name}: {len(image_files)} images")

        # Extract features for each image
        for image_path in sorted(image_files):
            # Construct output path maintaining location/body_part structure
            relative_path = image_path.relative_to(images_dir)
            output_path = features_dir / relative_path.with_suffix(".pt")

            # Skip if features already exist
            if output_path.exists():
                logging.debug(f"Features already exist for {image_path.name}, skipping")
                continue

            # Extract features
            try:
                feats = extract_sift_features(
                    image_path=image_path,
                    max_num_keypoints=max_num_keypoints,
                )

                # Save features
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(feats, output_path)

                num_kpts = feats["keypoints"].shape[0]
                logging.debug(f"Extracted features: {image_path.name} -> {num_kpts} keypoints")

                total_images += 1
                total_keypoints += num_kpts

            except Exception as e:
                logging.error(f"Failed to extract features from {image_path}: {e}")
                continue

    # Print summary
    avg_keypoints = total_keypoints / total_images if total_images > 0 else 0
    logging.info("\nFeature extraction complete!")
    logging.info(f"  Total images processed: {total_images}")
    logging.info(f"  Total keypoints extracted: {total_keypoints}")
    logging.info(f"  Average keypoints per image: {avg_keypoints:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract SIFT features from catalog images"
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        required=True,
        help="Path to catalog database directory",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default="sift",
        choices=["sift"],
        help="Feature extractor to use (default: sift)",
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=2048,
        help="Maximum number of keypoints to extract per image (default: 2048)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    extract_catalog_features(
        catalog_dir=args.catalog_dir,
        extractor_name=args.extractor,
        max_num_keypoints=args.max_keypoints,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
