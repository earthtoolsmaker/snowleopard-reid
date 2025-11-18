"""Split YOLO segmentation dataset into train/val/test sets with temporal anti-leakage.

This script creates reproducible train/val/test splits while preventing data leakage
between splits. It groups images from the same photoshoot session together to ensure
that related images (e.g., burst sequences, same lighting conditions) stay in the same
split, preserving the integrity of the evaluation process.

This script:
1. Loads curated selection JSON from FiftyOne export
2. Groups images by (location, individual) pairs
3. Detects photoshoot sessions using filename patterns
4. Randomly assigns complete sessions to train/val/test splits
5. Copies images and labels to output directory with renamed files
6. Generates YOLO data.yaml configuration file
7. Creates split_metadata.yaml for reproducibility tracking

Usage:
    python scripts/data/split_yolo_dataset.py --input-curated-json <path> --output-dir <path> [options]

Example:
    python scripts/data/split_yolo_dataset.py \
        --input-curated-json "./data/02_processed/fiftyone/yolo/segmentation/fiftyone_curated_selection.json" \
        --output-dir "./data/03_model_input/yolo/segmentation" \
        --train-ratio 0.8 \
        --val-ratio 0.1 \
        --test-ratio 0.1 \
        --random-seed 42

Notes:
    - Temporal anti-leakage prevents overfitting by keeping related images together.
      Without it, a model could "memorize" specific photoshoot conditions (lighting,
      angle, background) appearing in both train and test sets, inflating validation
      metrics while performing poorly on truly novel images.
    - Session detection uses three heuristics: date patterns (20141213_*),
      image sequences (IMG_4173_*), and alphanumeric sequences (360A0847).
    - Grouping by (location, individual) ensures all body parts of the same individual
      stay together in the same split, preventing leakage across body parts.
    - For small datasets with few sessions per (location, individual), splits may be
      uneven but will prioritize keeping sessions intact over exact ratio matching.
    - Random seed ensures reproducibility across runs for experiment tracking.
"""

import argparse
import json
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Split YOLO dataset with temporal anti-leakage"
    )
    parser.add_argument(
        "--input-curated-json",
        type=Path,
        required=True,
        help="Path to FiftyOne curated selection JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for train/val/test splits",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


def validate_split_ratios(train: float, val: float, test: float) -> None:
    """Validate that split ratios sum to 1.0."""
    total = train + val + test
    if not abs(total - 1.0) < 1e-6:
        msg = f"Split ratios must sum to 1.0, got {total:.6f} (train={train}, val={val}, test={test})"
        raise ValueError(msg)


def extract_session_key(filename: str) -> str:
    """
    Extract session identifier from filename for grouping related images.

    Session detection heuristics:
    1. Date patterns: 20141213_* → "20141213"
    2. Image sequences: IMG_4173_* → "IMG_4173" (base number)
    3. Numbered sequences: 360A0847 → "360A0847" (base number)

    Returns a key that groups images from the same photoshoot session.
    """
    # Pattern 1: Date-based filenames (20141213_IMG_5385.jpg)
    date_match = re.search(r"(\d{8})", filename)
    if date_match:
        return f"date_{date_match.group(1)}"

    # Pattern 2: IMG_XXXX style (IMG_4173_41F.jpg)
    img_match = re.match(r"(IMG_\d+)", filename)
    if img_match:
        # Group by base number (floor to nearest 10 for burst sequences)
        base_num = img_match.group(1)
        try:
            num = int(re.search(r"\d+", base_num).group())
            session_num = (num // 10) * 10  # Group sequences of 10
            return f"img_seq_{session_num}"
        except (AttributeError, ValueError):
            return f"img_{base_num}"

    # Pattern 3: Alphanumeric sequences (360A0847.jpg)
    alphanum_match = re.match(r"([A-Z0-9]+)", filename)
    if alphanum_match:
        base = alphanum_match.group(1)
        # Try to extract numeric part for grouping
        num_match = re.search(r"\d+", base)
        if num_match:
            try:
                num = int(num_match.group())
                session_num = (num // 10) * 10
                return f"seq_{base[:4]}_{session_num}"
            except ValueError:
                pass
        return f"alpha_{base}"

    # Fallback: use full filename (each image is its own session)
    return f"file_{filename}"


def group_into_sessions(
    samples: list[dict[str, Any]], location: str, individual: str
) -> list[list[dict[str, Any]]]:
    """
    Group images from the same location and individual into photoshoot sessions.

    Args:
        samples: List of sample dictionaries with image_path, label_path, etc.
        location: Geographic location
        individual: Individual snow leopard name

    Returns:
        List of sessions, where each session is a list of samples
    """
    # Group by session key
    session_groups = defaultdict(list)
    for sample in samples:
        filename = Path(sample["image_path"]).name
        session_key = extract_session_key(filename)
        session_groups[session_key].append(sample)

    # Convert to list of sessions
    sessions = list(session_groups.values())

    # Sort each session by filename for consistency
    for session in sessions:
        session.sort(key=lambda s: Path(s["image_path"]).name)

    # Sort sessions by first filename
    sessions.sort(key=lambda sess: Path(sess[0]["image_path"]).name)

    return sessions


def split_sessions(
    sessions: list[list[dict[str, Any]]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split sessions into train/val/test sets.

    Args:
        sessions: List of sessions (each session is a list of samples)
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for shuffling

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    import random

    # Shuffle sessions
    sessions_copy = sessions.copy()
    random.Random(random_seed).shuffle(sessions_copy)

    # Calculate split points based on number of sessions
    total_sessions = len(sessions_copy)
    train_count = max(1, int(total_sessions * train_ratio))
    val_count = max(0, int(total_sessions * val_ratio))

    # Adjust if needed (prioritize train > val > test)
    if val_count == 0 and total_sessions > 1:
        val_count = 1
        train_count -= 1

    # Split sessions
    train_sessions = sessions_copy[:train_count]
    val_sessions = sessions_copy[train_count : train_count + val_count]
    test_sessions = sessions_copy[train_count + val_count :]

    # Flatten sessions into sample lists
    train_samples = [sample for session in train_sessions for sample in session]
    val_samples = [sample for session in val_sessions for sample in session]
    test_samples = [sample for session in test_sessions for sample in session]

    return train_samples, val_samples, test_samples


def copy_files(
    samples: list[dict[str, Any]],
    output_dir: Path,
    split_name: str,
    base_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Copy image and label files to output directory.

    Args:
        samples: List of samples with image_path and label_path
        output_dir: Output base directory
        split_name: 'train', 'val', or 'test'
        base_dir: Base directory for resolving relative paths

    Returns:
        Tuple of (copied_samples, original_samples) where:
        - copied_samples: List with updated paths in output directory
        - original_samples: List with original source paths relative to data/
    """
    images_dir = output_dir / split_name / "images"
    labels_dir = output_dir / split_name / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    copied_samples = []
    original_samples = []

    for sample in samples:
        # Resolve paths
        image_path = Path(sample["image_path"])
        label_path = Path(sample["label_path"])

        if not image_path.is_absolute():
            image_path = base_dir / image_path
        if not label_path.is_absolute():
            label_path = base_dir / label_path

        # Store original path relative to project root (with data/ prefix)
        original_sample = sample.copy()
        # Remove leading ./ if present
        orig_img_str = str(sample["image_path"])
        if orig_img_str.startswith("./"):
            orig_img_str = orig_img_str[2:]
        # Ensure it starts with 'data/'
        if not orig_img_str.startswith("data/"):
            orig_img_str = f"data/{orig_img_str}"
        original_sample["image_path"] = orig_img_str
        original_samples.append(original_sample)

        # Create unique filename: location_individual_bodypart_filename
        location = sample["location"]
        individual = sample["individual"]
        body_part = sample["body_part"]
        image_filename = f"{location}_{individual}_{body_part}_{image_path.name}"
        label_filename = f"{location}_{individual}_{body_part}_{label_path.name}"

        # Copy files
        dest_image = images_dir / image_filename
        dest_label = labels_dir / label_filename

        shutil.copy2(image_path, dest_image)
        shutil.copy2(label_path, dest_label)

        # Update sample with new paths
        copied_sample = sample.copy()
        copied_sample["image_path"] = str(dest_image.relative_to(output_dir))
        copied_sample["label_path"] = str(dest_label.relative_to(output_dir))
        copied_samples.append(copied_sample)

    return copied_samples, original_samples


def generate_data_yaml(output_dir: Path) -> None:
    """
    Generate YOLO data.yaml configuration file.

    Args:
        output_dir: Output directory containing train/val/test
    """
    data_yaml = {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": ["snow_leopard"],
    }

    yaml_path = output_dir / "data.yaml"
    with yaml_path.open("w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    logger.info("Generated data.yaml with 1 class: snow_leopard")


def generate_split_metadata(
    output_dir: Path,
    train_samples: list[dict],
    val_samples: list[dict],
    test_samples: list[dict],
    args: argparse.Namespace,
) -> None:
    """
    Generate split metadata YAML for reproducibility.

    Args:
        output_dir: Output directory
        train_samples: Training samples with original image paths (relative to data/)
        val_samples: Validation samples with original image paths (relative to data/)
        test_samples: Test samples with original image paths (relative to data/)
        args: Command line arguments
    """
    metadata = {
        "split_config": {
            "random_seed": args.random_seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
        },
        "statistics": {
            "train_count": len(train_samples),
            "val_count": len(val_samples),
            "test_count": len(test_samples),
            "total_count": len(train_samples) + len(val_samples) + len(test_samples),
        },
        "splits": {
            "train": [s["image_path"] for s in train_samples],
            "val": [s["image_path"] for s in val_samples],
            "test": [s["image_path"] for s in test_samples],
        },
    }

    metadata_path = output_dir / "split_metadata.yaml"
    with metadata_path.open("w") as f:
        # Custom YAML dumper that only quotes paths in splits
        class QuotedDumper(yaml.SafeDumper):
            pass

        def str_representer(dumper, data):
            # Check if we're inside a 'splits' list by checking the context
            # Quote if it looks like a file path (contains 'data/')
            if "data/" in data:
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        QuotedDumper.add_representer(str, str_representer)

        yaml.dump(
            metadata,
            f,
            Dumper=QuotedDumper,
            default_flow_style=False,
            sort_keys=False,
            width=float("inf"),  # Prevent line wrapping
        )

    logger.info("Generated split_metadata.yaml")


def main() -> None:
    """Main execution function."""
    args = parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate split ratios
    validate_split_ratios(
        train=args.train_ratio, val=args.val_ratio, test=args.test_ratio
    )
    logger.info(
        f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}"
    )

    # Load curated selection
    logger.info(f"Loading curated selection from {args.input_curated_json}")
    with args.input_curated_json.open() as f:
        curated_data = json.load(f)

    selected_samples = curated_data["selected_samples"]
    logger.info(f"Loaded {len(selected_samples)} selected samples")

    # Get base directory for resolving relative paths
    # Paths in JSON are relative to project root (start with ./data/...)
    # So base_dir should be the project root
    base_dir = Path.cwd()

    # Group samples by (location, individual)
    groups = defaultdict(list)
    for sample in selected_samples:
        key = (sample["location"], sample["individual"])
        groups[key].append(sample)

    logger.info(f"Found {len(groups)} unique (location, individual) groups")

    # Split each group
    all_train_samples = []
    all_val_samples = []
    all_test_samples = []

    for (location, individual), samples in sorted(groups.items()):
        logger.debug(f"Processing {location}/{individual}: {len(samples)} images")

        # Group into sessions
        sessions = group_into_sessions(
            samples=samples, location=location, individual=individual
        )
        logger.debug(f"  Found {len(sessions)} sessions")

        # Split sessions
        train_samples, val_samples, test_samples = split_sessions(
            sessions=sessions,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed,
        )

        logger.debug(
            f"  Split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
        )

        all_train_samples.extend(train_samples)
        all_val_samples.extend(val_samples)
        all_test_samples.extend(test_samples)

    # Log overall statistics
    total = len(all_train_samples) + len(all_val_samples) + len(all_test_samples)
    logger.info("\nOverall split statistics:")
    logger.info(
        f"  Train: {len(all_train_samples)} ({len(all_train_samples) / total * 100:.1f}%)"
    )
    logger.info(
        f"  Val:   {len(all_val_samples)} ({len(all_val_samples) / total * 100:.1f}%)"
    )
    logger.info(
        f"  Test:  {len(all_test_samples)} ({len(all_test_samples) / total * 100:.1f}%)"
    )
    logger.info(f"  Total: {total}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    logger.info("\nCopying files...")
    copied_train, original_train = copy_files(
        samples=all_train_samples,
        output_dir=args.output_dir,
        split_name="train",
        base_dir=base_dir,
    )
    copied_val, original_val = copy_files(
        samples=all_val_samples,
        output_dir=args.output_dir,
        split_name="val",
        base_dir=base_dir,
    )
    copied_test, original_test = copy_files(
        samples=all_test_samples,
        output_dir=args.output_dir,
        split_name="test",
        base_dir=base_dir,
    )

    # Generate data.yaml
    logger.info("\nGenerating configuration files...")
    generate_data_yaml(args.output_dir)

    # Generate split metadata with original paths
    generate_split_metadata(
        output_dir=args.output_dir,
        train_samples=original_train,
        val_samples=original_val,
        test_samples=original_test,
        args=args,
    )

    logger.info("\n✓ Dataset split complete!")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info("  Classes: 1 class (snow_leopard)")


if __name__ == "__main__":
    main()
