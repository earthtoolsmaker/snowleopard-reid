"""
Parse individuals_review.md and generate YAML config files.

Extracts canonical individual names and mapping rules from the review file
and creates separate YAML files for individuals and mappings.
"""

import re
import yaml
from pathlib import Path
from collections import defaultdict


def parse_review_file(review_file: Path) -> dict:
    """
    Parse the review markdown file.

    Returns:
        {
            'naryn': {
                'individuals': [...],
                'mappings': {canonical_name: [variants...]}
            },
            'sarychat': {...}
        }
    """
    config = {
        "naryn": {"individuals": [], "mappings": defaultdict(list)},
        "sarychat": {"individuals": [], "mappings": defaultdict(list)},
    }

    current_location = None

    with open(review_file, "r") as f:
        for line in f:
            # Check for location headers
            if line.startswith("## NARYN"):
                current_location = "naryn"
                continue
            elif line.startswith("## SARYCHAT"):
                current_location = "sarychat"
                continue

            # Skip non-table lines
            if not line.startswith("|") or current_location is None:
                continue

            # Skip header and separator rows
            if "---" in line or "Individual Name" in line:
                continue

            # Parse table row
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue

            individual_name = parts[1]
            action_text = parts[3]

            if not individual_name or not action_text:
                continue

            action_text = action_text.strip()
            if not action_text:
                continue

            action_lower = action_text.lower()

            if action_lower == "keep":
                # This is a canonical individual
                config[current_location]["individuals"].append(individual_name)

            elif action_lower == "remove":
                # Skip removed individuals
                continue

            elif "->" in action_text:
                # This is a mapping: source -> target
                match = re.search(
                    r"(?:KEEP|MERGE)\s*->\s*(\S+)", action_text, re.IGNORECASE
                )
                if match:
                    target = match.group(1).strip()

                    # Only add mapping if source != target (avoid self-referencing)
                    if individual_name != target:
                        config[current_location]["mappings"][target].append(
                            individual_name
                        )

                    # Make sure target is in individuals list
                    if target not in config[current_location]["individuals"]:
                        config[current_location]["individuals"].append(target)

    # Clean up and finalize
    for location in config:
        # Remove duplicates from individuals list while preserving order
        seen = set()
        unique_individuals = []
        for ind in config[location]["individuals"]:
            if ind not in seen:
                seen.add(ind)
                unique_individuals.append(ind)

        # Sort alphabetically
        config[location]["individuals"] = sorted(unique_individuals)

        # Convert mappings to regular dict
        config[location]["mappings"] = dict(config[location]["mappings"])

    return config


def create_yaml_configs(config: dict, output_base_dir: Path):
    """
    Create YAML config files for each location.

    Creates:
        - {output_base_dir}/{location}/individuals.yaml
        - {output_base_dir}/{location}/mappings.yaml
    """
    for location, data in config.items():
        location_dir = output_base_dir / location
        location_dir.mkdir(parents=True, exist_ok=True)

        # Write individuals.yaml
        individuals_file = location_dir / "individuals.yaml"
        with open(individuals_file, "w") as f:
            yaml.dump(
                {"individuals": data["individuals"]},
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        print(f"✓ Created {individuals_file}")
        print(f"  {len(data['individuals'])} canonical individuals")

        # Write mappings.yaml
        mappings_file = location_dir / "mappings.yaml"
        with open(mappings_file, "w") as f:
            yaml.dump(
                {"mappings": data["mappings"]},
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        print(f"✓ Created {mappings_file}")
        print(f"  {len(data['mappings'])} mapping rules")
        print()


def main():
    review_file = Path("individuals_review.md")
    output_base_dir = Path("data/02_processed/config")

    if not review_file.exists():
        print(f"❌ Error: Review file not found: {review_file}")
        return 1

    print(f"Parsing {review_file}...")
    config = parse_review_file(review_file)

    print(f"\nCreating YAML configs in {output_base_dir}/...")
    create_yaml_configs(config, output_base_dir)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for location in ["naryn", "sarychat"]:
        print(f"\n{location.upper()}:")
        print(f"  Canonical individuals: {len(config[location]['individuals'])}")
        print(f"  Mapping rules: {len(config[location]['mappings'])}")
        total_variants = sum(
            len(variants) for variants in config[location]["mappings"].values()
        )
        print(f"  Total variants mapped: {total_variants}")

    return 0


if __name__ == "__main__":
    exit(main())
