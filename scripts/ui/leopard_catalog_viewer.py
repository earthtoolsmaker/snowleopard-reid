"""Snow Leopard Catalog Viewer - Gradio UI for browsing the leopard catalog.

This is a simplified viewer for exploring the snow leopard identification catalog.
It displays all individuals with their reference images organized by location and body part.

Usage:
    python scripts/ui/leopard_catalog_viewer.py [--catalog-root PATH] [--port PORT]

Example:
    python scripts/ui/leopard_catalog_viewer.py \
        --catalog-root ./data/08_catalog/v1.0 \
        --port 7860
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import yaml
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Configuration for the catalog viewer application."""

    catalog_root: Path
    port: int = 7860


def load_catalog_index(catalog_root: Path) -> dict:
    """Load the catalog index YAML file.

    Args:
        catalog_root: Root directory of the catalog

    Returns:
        Dictionary containing catalog index data
    """
    index_path = catalog_root / "catalog_index.yaml"
    if not index_path.exists():
        raise FileNotFoundError(f"Catalog index not found: {index_path}")

    with open(index_path, "r") as f:
        return yaml.safe_load(f)


def load_individual_metadata(catalog_root: Path, metadata_path: str) -> dict:
    """Load metadata for a specific individual.

    Args:
        catalog_root: Root directory of the catalog
        metadata_path: Relative path to individual's metadata file

    Returns:
        Dictionary containing individual metadata
    """
    full_path = catalog_root / metadata_path
    if not full_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {full_path}")

    with open(full_path, "r") as f:
        return yaml.safe_load(f)


def create_leopard_tab(leopard_metadata: Dict, config: AppConfig):
    """Create a tab for displaying a single leopard's images.

    Args:
        leopard_metadata: Metadata dictionary for the leopard
        config: Application configuration
    """
    leopard_name = leopard_metadata["individual_name"]
    location = leopard_metadata["location"]
    total_images = leopard_metadata["statistics"]["total_reference_images"]
    body_parts = leopard_metadata["statistics"]["body_parts_represented"]

    with gr.Tab(f"{leopard_name}"):
        # Header with statistics
        gr.Markdown(
            f"### {leopard_name.title()}\n"
            f"**Location:** {location} | "
            f"**{total_images} images** | "
            f"**Body parts:** {', '.join(body_parts)}"
        )

        # Load all images with captions
        gallery_data = []
        for img_entry in leopard_metadata["reference_images"]:
            img_path = config.catalog_root / "database" / img_entry["path"]
            body_part = img_entry["body_part"]

            try:
                img = Image.open(img_path)
                # Caption format: "location - body_part"
                caption = f"{location} - {body_part}"
                gallery_data.append((img, caption))
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")

        # Display gallery
        gr.Gallery(
            value=gallery_data,
            label=f"Reference Images for {leopard_name.title()}",
            columns=6,
            height=700,
            object_fit="scale-down",
            allow_preview=True,
        )


def create_catalog_browser_tab(config: AppConfig):
    """Create the catalog browser tab with all individuals.

    Args:
        config: Application configuration
    """
    with gr.Tab("📚 Explore Catalog"):
        gr.Markdown(
            """
            ## Snow Leopard Catalog Browser

            Browse the reference catalog of known snow leopard individuals.
            Each individual has multiple reference images from different body parts and locations.
            """
        )

        # Load catalog data
        try:
            catalog_index = load_catalog_index(config.catalog_root)

            # Display catalog statistics
            stats = catalog_index.get("statistics", {})
            gr.Markdown(
                f"""
                ### Catalog Statistics
                - **Total Individuals:** {stats.get("total_individuals", "N/A")}
                - **Total Images:** {stats.get("total_reference_images", "N/A")}
                - **Locations:** {", ".join(stats.get("locations", []))}
                - **Body Parts:** {", ".join(stats.get("body_parts", []))}
                """
            )

            gr.Markdown("---")
            gr.Markdown("### Individual Leopards by Location")

            # Group individuals by location
            individuals_by_location = {}
            for individual_entry in catalog_index["individuals"]:
                location = individual_entry["location"]
                if location not in individuals_by_location:
                    individuals_by_location[location] = []
                individuals_by_location[location].append(individual_entry)

            # Create tabs for each location
            with gr.Tabs():
                for location in sorted(individuals_by_location.keys()):
                    with gr.Tab(f"📍 {location.title()}"):
                        # Create subtabs for each individual in this location
                        with gr.Tabs():
                            for individual_entry in individuals_by_location[location]:
                                # Load individual metadata
                                metadata_path = individual_entry["metadata_path"]
                                leopard_metadata = load_individual_metadata(
                                    config.catalog_root, metadata_path
                                )

                                # Create tab for this individual
                                create_leopard_tab(leopard_metadata, config)

        except Exception as e:
            logger.error(f"Error loading catalog: {e}")
            gr.Markdown(f"⚠️ **Error loading catalog:** {e}")


def create_app(config: AppConfig) -> gr.Blocks:
    """Create the Gradio application.

    Args:
        config: Application configuration

    Returns:
        Gradio Blocks application
    """
    with gr.Blocks(
        title="Snow Leopard Catalog Viewer",
        theme=gr.themes.Soft(),
    ) as app:
        # Header
        gr.Markdown(
            """
            # 🐆 Snow Leopard Catalog Viewer

            Explore the snow leopard identification catalog with reference images organized by individual, location, and body part.
            """
        )

        # Main catalog browser tab
        create_catalog_browser_tab(config)

        # Footer
        gr.Markdown(
            """
            ---
            **Snow Leopard Re-Identification Project** | Catalog Version: 1.0.0
            """
        )

    return app


def main():
    """Main entry point for the catalog viewer."""
    parser = argparse.ArgumentParser(
        description="Snow Leopard Catalog Viewer - Browse the leopard identification catalog"
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=Path("data/08_catalog/v1.0"),
        help="Path to catalog root directory (default: data/08_catalog/v1.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the web interface on (default: 7860)",
    )

    args = parser.parse_args()

    # Validate catalog exists
    if not args.catalog_root.exists():
        logger.error(f"Catalog directory not found: {args.catalog_root}")
        logger.error("Please ensure you have built the catalog first.")
        return

    # Create config
    config = AppConfig(
        catalog_root=args.catalog_root,
        port=args.port,
    )

    logger.info(f"Loading catalog from: {config.catalog_root}")
    logger.info(f"Starting web interface on port {config.port}")

    # Create and launch app
    app = create_app(config)
    app.launch(
        server_name="0.0.0.0",
        server_port=config.port,
        share=False,
    )


if __name__ == "__main__":
    main()
