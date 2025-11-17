"""
Reorganize snow leopard images using canonical individual lists and mappings.

Uses YAML config files to map image filenames to canonical individual names.
"""

import re
import shutil
import yaml
from pathlib import Path
from collections import defaultdict
import argparse


# Body part folder mappings
BODY_PART_MAPPING = {
    "Head": "head",
    "Left flank": "left_flank",
    "Right flank": "right_flank",
    "Tail": "tail",
    "sexe blessure autres": "misc",
}

# Image extensions to process
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def load_config(config_dir: Path) -> tuple[list[str], dict[str, list[str]]]:
    """
    Load individuals and mappings from YAML config files.

    Args:
        config_dir: Path to config directory (e.g., data/02_processed/config/naryn/)

    Returns:
        (individuals_list, mappings_dict)
    """
    individuals_file = config_dir / "individuals.yaml"
    mappings_file = config_dir / "mappings.yaml"

    with open(individuals_file, "r") as f:
        individuals_data = yaml.safe_load(f)
        individuals = individuals_data.get("individuals", [])

    with open(mappings_file, "r") as f:
        mappings_data = yaml.safe_load(f)
        mappings = mappings_data.get("mappings", {})

    return individuals, mappings


def extract_name_from_filename(filename: str) -> str:
    """
    Extract the individual name part from filename.

    Removes common prefixes but keeps the meaningful name part.
    """
    name_part = Path(filename).stem

    # Common prefixes to remove
    prefixes = [
        r"^\d{8}\s+",  # 01060311 Naguima
        r"^IMG_\d+\s+",  # IMG_1634 Ilep
        r"^EK\d+\s+M?\d*\s*",  # EK000697 M3 Zhyldyz
        r"^Cdy\d+\s+",  # Cdy00020 Sogusker
        r"^IMAG\d+\s+",  # IMAG2724 Muktar
        r"^Birbaital\s+",  # Birbaital Italya 04040553
        r"^\d{4,}\s+",  # Generic number prefix
    ]

    for prefix_pattern in prefixes:
        name_part = re.sub(prefix_pattern, "", name_part, flags=re.IGNORECASE)

    # Remove leading numbered variants like "(3) ayima" -> "ayima"
    name_part = re.sub(r"^\(\d+\)\s+", "", name_part)

    # Remove trailing dates and variant numbers
    name_part = re.sub(r"\s+\d{8}$", "", name_part)
    name_part = re.sub(r"\s*\(\d+\)$", "", name_part)

    # Remove " - copie" suffixes
    name_part = re.sub(
        r"\s*-\s*copie(\s*-\s*copie)?$", "", name_part, flags=re.IGNORECASE
    )

    # Remove trailing location/side indicators like "à d" (à droite = to the right)
    name_part = re.sub(r"\s+à\s+[dg]$", "", name_part, flags=re.IGNORECASE)

    return name_part.strip().lower()


def match_to_canonical(
    extracted_name: str, individuals: list[str], mappings: dict[str, list[str]]
) -> str | None:
    """
    Match extracted name to a canonical individual name.

    Args:
        extracted_name: Name extracted from filename
        individuals: List of canonical individual names
        mappings: Dict mapping canonical names to variant lists

    Returns:
        Canonical individual name or None if no match
    """
    extracted_lower = extracted_name.lower()

    # First try exact match with canonical names
    for individual in individuals:
        if individual.lower() == extracted_lower:
            return individual

    # Then try matching against mapping variants
    for canonical_name, variants in mappings.items():
        for variant in variants:
            if variant.lower() == extracted_lower:
                return canonical_name

    # Try partial matching if name is contained in variant
    for canonical_name, variants in mappings.items():
        for variant in variants:
            if extracted_lower in variant.lower() or variant.lower() in extracted_lower:
                return canonical_name

    return None


def should_skip_file(filepath: Path) -> tuple[bool, str]:
    """
    Determine if a file should be skipped.

    Returns (should_skip, reason)
    """
    filename = filepath.name.lower()

    # Skip if contains "cub"
    if "cub" in filename:
        return True, "contains 'cub'"

    # Skip if not an image
    if filepath.suffix not in IMAGE_EXTENSIONS:
        return True, f"not an image (extension: {filepath.suffix})"

    # Skip Mac metadata files
    if "/__MACOSX/" in str(filepath) or filepath.name.startswith("._"):
        return True, "Mac metadata file"

    return False, ""


def reorganize_data(
    input_dir: Path, output_dir: Path, config_base_dir: Path, dry_run: bool = False
) -> dict:
    """
    Reorganize snow leopard images using canonical individual configs.

    Args:
        input_dir: Path to data/01_raw/locations/
        output_dir: Path to data/02_processed/locations/
        config_base_dir: Path to data/02_processed/config/
        dry_run: If True, don't copy files, just report what would be done

    Returns:
        Dictionary with processing statistics
    """
    stats = {
        "total_files_scanned": 0,
        "files_skipped": defaultdict(int),
        "files_copied": 0,
        "files_unmatched": 0,
        "unmatched_names": defaultdict(int),
        "individuals_found": defaultdict(lambda: defaultdict(lambda: defaultdict(int))),
        "errors": [],
    }

    # Process each location
    for location_dir in sorted(input_dir.iterdir()):
        if not location_dir.is_dir():
            continue

        location_name = location_dir.name.lower()
        print(f"\nProcessing location: {location_name}")

        # Load config for this location
        config_dir = config_base_dir / location_name
        if not config_dir.exists():
            print(f"  ⚠️  Config directory not found: {config_dir}")
            continue

        try:
            individuals, mappings = load_config(config_dir)
            print(f"  Loaded {len(individuals)} canonical individuals")
            print(f"  Loaded {len(mappings)} mapping rules")
        except Exception as e:
            error_msg = f"Error loading config for {location_name}: {e}"
            stats["errors"].append(error_msg)
            print(f"  ❌ {error_msg}")
            continue

        # Process all body part folders
        for body_part_dir in location_dir.iterdir():
            if not body_part_dir.is_dir():
                continue

            body_part_original = body_part_dir.name
            body_part_mapped = BODY_PART_MAPPING.get(body_part_original)

            if body_part_mapped is None:
                # Skip unmapped folders (like profiles)
                continue

            print(f"  Scanning {body_part_original}/ → {body_part_mapped}/")

            # Process all files
            for filepath in body_part_dir.iterdir():
                if not filepath.is_file():
                    continue

                stats["total_files_scanned"] += 1

                # Check if should skip
                should_skip, skip_reason = should_skip_file(filepath)
                if should_skip:
                    stats["files_skipped"][skip_reason] += 1
                    continue

                # Extract name from filename
                extracted_name = extract_name_from_filename(filepath.name)
                if not extracted_name:
                    stats["files_unmatched"] += 1
                    stats["unmatched_names"][filepath.name] += 1
                    continue

                # Match to canonical individual
                canonical_name = match_to_canonical(
                    extracted_name, individuals, mappings
                )
                if canonical_name is None:
                    stats["files_unmatched"] += 1
                    stats["unmatched_names"][extracted_name] += 1
                    continue

                # Build output path
                output_path = (
                    output_dir
                    / location_name
                    / canonical_name
                    / body_part_mapped
                    / filepath.name
                )

                # Copy file
                if not dry_run:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(filepath, output_path)
                    except Exception as e:
                        error_msg = f"Error copying {filepath} to {output_path}: {e}"
                        stats["errors"].append(error_msg)
                        print(f"    ❌ {error_msg}")
                        continue

                stats["files_copied"] += 1
                stats["individuals_found"][location_name][canonical_name][
                    body_part_mapped
                ] += 1

    return stats


def print_summary(stats: dict):
    """Print summary of processing results."""
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)

    print(f"\nTotal files scanned: {stats['total_files_scanned']}")
    print(f"Files copied: {stats['files_copied']}")
    print(f"Files unmatched: {stats['files_unmatched']}")

    if stats["files_skipped"]:
        print("\nFiles skipped by reason:")
        for reason, count in sorted(stats["files_skipped"].items()):
            print(f"  - {reason}: {count}")

    if stats["unmatched_names"]:
        print("\nUnmatched names (showing top 20):")
        sorted_unmatched = sorted(
            stats["unmatched_names"].items(), key=lambda x: x[1], reverse=True
        )
        for name, count in sorted_unmatched[:20]:
            print(f"  - {name}: {count} files")
        if len(sorted_unmatched) > 20:
            print(f"  ... and {len(sorted_unmatched) - 20} more")

    if stats["errors"]:
        print(f"\n⚠️  Errors encountered: {len(stats['errors'])}")
        for error in stats["errors"][:10]:
            print(f"  - {error}")
        if len(stats["errors"]) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")

    print("\n" + "-" * 70)
    print("INDIVIDUALS BY LOCATION")
    print("-" * 70)

    for location in sorted(stats["individuals_found"].keys()):
        individuals = stats["individuals_found"][location]
        print(f"\n{location.upper()} ({len(individuals)} individuals):")

        for individual in sorted(individuals.keys()):
            body_parts = individuals[individual]
            total_images = sum(body_parts.values())
            parts_str = ", ".join(
                f"{bp}: {count}" for bp, count in sorted(body_parts.items())
            )
            print(f"  {individual}: {total_images} images ({parts_str})")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize snow leopard images using canonical individual configs"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/01_raw/locations"),
        help="Input directory (default: data/01_raw/locations)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/02_processed/locations"),
        help="Output directory (default: data/02_processed/locations)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("data/02_processed/config"),
        help="Config directory (default: data/02_processed/config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't copy files, just show what would be done",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input_dir.exists():
        print(f"❌ Error: Input directory does not exist: {args.input_dir}")
        return 1

    if not args.config_dir.exists():
        print(f"❌ Error: Config directory does not exist: {args.config_dir}")
        return 1

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Config directory: {args.config_dir}")

    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No files will be copied\n")

    # Run reorganization
    stats = reorganize_data(
        args.input_dir, args.output_dir, args.config_dir, dry_run=args.dry_run
    )

    # Print summary
    print_summary(stats)

    return 0


if __name__ == "__main__":
    exit(main())
