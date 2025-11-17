"""Build catalog database from selection.yaml.

This script reads the selection.yaml file and builds the catalog database structure
by copying selected images into the organized catalog directory structure.

The catalog structure is:
database/
  {individual}/
    images/
      {location}/
        {body_part}/
          *.jpg
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict

import yaml


def build_catalog_structure(
    selections: Dict, catalog_dir: Path, dry_run: bool = False
) -> None:
    """Build catalog directory structure and copy images.

    Args:
        selections: Dictionary with structure {location: {individual: {body_part: [paths]}}}
        catalog_dir: Path to catalog database directory
        dry_run: If True, only print actions without copying files
    """
    stats = {
        "individuals": set(),
        "locations": set(),
        "body_parts": set(),
        "images_copied": 0,
    }

    # Iterate through the selection structure
    for location, individuals in selections.items():
        stats["locations"].add(location)

        for individual, body_parts in individuals.items():
            stats["individuals"].add(individual)

            # Create individual directory with location first
            individual_dir = catalog_dir / location / individual / "images"
            if not dry_run:
                individual_dir.mkdir(parents=True, exist_ok=True)
            else:
                print(f"Would create directory: {individual_dir}")

            for body_part, image_paths in body_parts.items():
                stats["body_parts"].add(body_part)

                # Create body part directory
                body_part_dir = individual_dir / body_part
                if not dry_run:
                    body_part_dir.mkdir(parents=True, exist_ok=True)
                else:
                    print(f"Would create directory: {body_part_dir}")

                # Copy each image
                for image_path in image_paths:
                    src_path = Path(image_path)
                    if not src_path.exists():
                        print(f"Warning: Source image not found: {src_path}")
                        continue

                    # Destination path
                    dst_path = body_part_dir / src_path.name

                    if dry_run:
                        print(f"Would copy: {src_path} -> {dst_path}")
                    else:
                        shutil.copy2(src_path, dst_path)
                        stats["images_copied"] += 1

    # Print statistics
    print("\nCatalog build complete!")
    print(f"  Individuals: {len(stats['individuals'])}")
    print(f"  Locations: {sorted(stats['locations'])}")
    print(f"  Body parts: {sorted(stats['body_parts'])}")
    if not dry_run:
        print(f"  Images copied: {stats['images_copied']}")


def main():
    parser = argparse.ArgumentParser(
        description="Build catalog database from selection.yaml"
    )
    parser.add_argument(
        "--selection-file",
        type=Path,
        required=True,
        help="Path to selection.yaml file",
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        required=True,
        help="Path to catalog database directory (will be created if needed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without copying files",
    )

    args = parser.parse_args()

    # Validate selection file
    if not args.selection_file.exists():
        raise FileNotFoundError(f"Selection file not found: {args.selection_file}")

    # Load selection.yaml
    print(f"Loading selection file: {args.selection_file}")
    with open(args.selection_file, "r") as f:
        data = yaml.safe_load(f)

    selections = data.get("selections", {})
    if not selections:
        raise ValueError("No selections found in selection.yaml")

    print(
        f"{'DRY RUN - ' if args.dry_run else ''}Building catalog at: {args.catalog_dir}"
    )

    # Build catalog structure
    build_catalog_structure(selections, args.catalog_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
