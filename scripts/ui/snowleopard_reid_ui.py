"""Gradio web application for snow leopard identification and catalog exploration.

This interactive web interface provides an easy-to-use frontend for the snow
leopard identification system. Users can upload images, view matches against the catalog,
and explore reference leopards through a browser-based UI powered by Gradio.

Features:
- Upload snow leopard images or select from examples
- Run full identification pipeline with interactive progress
- View top-K matches with Wasserstein distance scores
- Explore complete leopard catalog with thumbnails
- Visualize matched keypoints between query and catalog images

Usage:
    python scripts/ui/snowleopard_reid_ui.py [options]

Example:
    python scripts/ui/snowleopard_reid_ui.py \
        --catalog-dir ./data/08_catalog/v1.0 \
        --model-path ./data/04_models/yolo/segmentation/best/weights/best.pt \
        --port 7860 \
        --share

    # Or use the make command:
    make ui

Notes:
    - Default port is 7860, accessible at http://localhost:7860
    - Use --share to create public URL (Gradio tunnel)
    - Requires catalog with extracted features (run extract_catalog_features.py first)
    - GPU automatically used when available for faster inference
"""

import argparse
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import yaml
from PIL import Image

from snowleopard_reid.catalog import (
    get_available_body_parts,
    get_available_locations,
    get_catalog_metadata_for_id,
    load_catalog_index,
    load_leopard_metadata,
)
from snowleopard_reid.pipeline.stages import (
    run_feature_extraction_stage,
    run_matching_stage,
    run_preprocess_stage,
    run_segmentation_stage,
    select_best_mask,
)
from snowleopard_reid.visualization import (
    draw_keypoints_overlay,
    draw_matched_keypoints,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Configuration for the Snow Leopard ID UI application."""

    model_path: Path
    catalog_root: Path
    examples_dir: Path
    top_k: int
    port: int
    share: bool


def get_available_extractors(catalog_root: Path) -> list[str]:
    """Get list of available feature extractors from catalog.

    Args:
        catalog_root: Root directory of the leopard catalog

    Returns:
        List of available extractor names (e.g., ['sift', 'superpoint'])
    """
    try:
        catalog_index = load_catalog_index(catalog_root)
        extractors = list(catalog_index.get("feature_extractors", {}).keys())
        if not extractors:
            logger.warning(f"No extractors found in catalog at {catalog_root}")
            return ["sift"]  # Default fallback
        return extractors
    except Exception as e:
        logger.error(f"Failed to load catalog index: {e}")
        return ["sift"]  # Default fallback


def parse_args() -> AppConfig:
    """Parse command-line arguments and return application configuration.

    Returns:
        AppConfig object with all configuration parameters
    """
    parser = argparse.ArgumentParser(
        description="Launch Snow Leopard ID UI for identification and catalog exploration"
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/04_models/yolo/segmentation/best/weights/best.pt"),
        help="Path to YOLO segmentation model weights (default: %(default)s)",
    )

    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=Path("data/08_catalog/v1.0"),
        help="Root directory of the leopard catalog (default: %(default)s)",
    )

    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=Path("data/07_external/ui/examples"),
        help="Directory containing example images (default: %(default)s)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top matches to return (default: %(default)s)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: %(default)s)",
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio tunnel URL",
    )

    args = parser.parse_args()

    return AppConfig(
        model_path=args.model_path,
        catalog_root=args.catalog_root,
        examples_dir=args.examples_dir,
        top_k=args.top_k,
        port=args.port,
        share=args.share,
    )


# Global state for models and catalog (loaded at startup)
LOADED_MODELS = {}


def load_catalog_data(config: AppConfig):
    """Load catalog index and individual leopard metadata.

    Args:
        config: Application configuration containing catalog_root

    Returns:
        Tuple of (catalog_index, individuals_data)
    """
    catalog_index_path = config.catalog_root / "catalog_index.yaml"

    # Load catalog index
    with open(catalog_index_path) as f:
        catalog_index = yaml.safe_load(f)

    # Load metadata for each individual
    individuals_data = []
    for individual in catalog_index["individuals"]:
        metadata_path = config.catalog_root / individual["metadata_path"]
        with open(metadata_path) as f:
            leopard_metadata = yaml.safe_load(f)
        individuals_data.append(leopard_metadata)

    return catalog_index, individuals_data


def initialize_models(config: AppConfig):
    """Load models at startup for faster inference.

    Args:
        config: Application configuration containing model_path
    """
    logger.info("Initializing models...")

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")

    # Load YOLO model once at startup
    from ultralytics import YOLO

    logger.info(f"Loading YOLO model from {config.model_path}")
    yolo_model = YOLO(str(config.model_path))
    LOADED_MODELS["yolo"] = yolo_model
    logger.info("YOLO model loaded successfully")

    # Store device info and catalog root for callbacks
    LOADED_MODELS["device"] = device
    LOADED_MODELS["catalog_root"] = config.catalog_root

    logger.info("Models initialized successfully")


def run_identification(
    image,
    extractor: str,
    selected_locations: list[str],
    selected_body_parts: list[str],
    config: AppConfig,
):
    """Run snow leopard identification pipeline on uploaded image.

    Args:
        image: PIL Image from Gradio upload
        extractor: Feature extractor to use ('sift', 'superpoint', 'disk', 'aliked')
        selected_locations: List of selected locations (includes "all" for no filtering)
        selected_body_parts: List of selected body parts (includes "all" for no filtering)
        config: Application configuration

    Returns:
        Tuple of UI components to update
    """
    if image is None:
        return (
            "⚠️ Please upload an image first",
            None,
            None,
            None,
            [],
            gr.update(visible=False),
        )

    # Convert filter selections to None if "all" is selected
    filter_locations = (
        None
        if not selected_locations or "all" in selected_locations
        else selected_locations
    )
    filter_body_parts = (
        None
        if not selected_body_parts or "all" in selected_body_parts
        else selected_body_parts
    )

    # Log applied filters
    if filter_locations or filter_body_parts:
        filter_desc = []
        if filter_locations:
            filter_desc.append(f"locations: {', '.join(filter_locations)}")
        if filter_body_parts:
            filter_desc.append(f"body parts: {', '.join(filter_body_parts)}")
        logger.info(f"Applied filters - {' | '.join(filter_desc)}")
    else:
        logger.info("No filters applied - matching against entire catalog")

    try:
        # Create temporary directory for this query
        temp_dir = Path(tempfile.mkdtemp(prefix="snowleopard_id_"))
        temp_image_path = temp_dir / "query.jpg"

        # Save uploaded image
        logger.info(f"Image type: {type(image)}")
        logger.info(f"Image mode: {image.mode if hasattr(image, 'mode') else 'N/A'}")
        logger.info(f"Image size: {image.size if hasattr(image, 'size') else 'N/A'}")
        image.save(temp_image_path, quality=95)

        # Verify saved image
        saved_size = temp_image_path.stat().st_size
        logger.info(f"Saved image size: {saved_size / 1024 / 1024:.2f} MB")

        logger.info(f"Processing query image: {temp_image_path}")

        device = LOADED_MODELS.get("device", "cpu")

        # Step 1: Run YOLO segmentation using pre-loaded model
        logger.info("Running YOLO segmentation...")
        yolo_model = LOADED_MODELS.get("yolo")

        seg_stage = run_segmentation_stage(
            model=yolo_model,
            image_path=temp_image_path,
            confidence_threshold=0.2,
            device=device,
        )

        predictions = seg_stage["data"]["predictions"]
        logger.info(f"Number of predictions: {len(predictions)}")

        if not predictions:
            logger.warning("No predictions found from YOLO")
            logger.warning(f"Full segmentation stage: {seg_stage}")
            return (
                "❌ No snow leopards detected in image",
                None,
                None,
                None,
                [],
                gr.update(visible=False),
            )

        # Step 2: Select best mask
        logger.info("Selecting best mask...")
        selected_idx, selected_pred = select_best_mask(
            predictions,
            strategy="confidence_area",
        )

        # Step 3: Preprocess (crop and mask)
        logger.info("Preprocessing query image...")
        prep_stage = run_preprocess_stage(
            image_path=temp_image_path,
            mask=selected_pred["mask"],
            padding=5,
        )

        cropped_image_pil = prep_stage["data"]["cropped_image"]

        # Save cropped image for visualization later
        cropped_path = temp_dir / "cropped.jpg"
        cropped_image_pil.save(cropped_path)

        # Step 4: Extract features
        logger.info(f"Extracting features using {extractor.upper()}...")
        feat_stage = run_feature_extraction_stage(
            image=cropped_image_pil,
            extractor=extractor,
            max_keypoints=2048,
            device=device,
        )

        query_features = feat_stage["data"]["features"]

        # Step 5: Match against catalog
        logger.info("Matching against catalog...")
        pairwise_dir = temp_dir / "pairwise"
        pairwise_dir.mkdir(exist_ok=True)

        match_stage = run_matching_stage(
            query_features=query_features,
            catalog_path=config.catalog_root,
            top_k=config.top_k,
            extractor=extractor,
            device=device,
            query_image_path=str(cropped_path),
            pairwise_output_dir=pairwise_dir,
            filter_locations=filter_locations,
            filter_body_parts=filter_body_parts,
        )

        matches = match_stage["data"]["matches"]

        if not matches:
            return (
                "❌ No matches found in catalog",
                None,
                cropped_image_pil,
                None,
                [],
                gr.update(visible=False),
            )

        # Top match
        top_match = matches[0]
        top_leopard_name = top_match["leopard_name"]
        top_wasserstein = top_match["wasserstein"]

        # Determine if it's a confident match (higher Wasserstein = better)
        if top_wasserstein > 0.2:
            confidence_indicator = "✅"
        elif top_wasserstein > 0.1:
            confidence_indicator = "⚠️"
        else:
            confidence_indicator = "❓"

        result_text = f"## {confidence_indicator} {top_leopard_name}"

        # Create segmentation visualization
        seg_viz = create_segmentation_viz(
            image_path=temp_image_path, mask=selected_pred["mask"]
        )

        # Generate extracted keypoints visualization
        extracted_kpts_viz = None
        try:
            # Extract keypoints from query features for visualization
            query_kpts = query_features["keypoints"].cpu().numpy()
            extracted_kpts_viz = draw_keypoints_overlay(
                image_path=cropped_path,
                keypoints=query_kpts,
                max_keypoints=500,
                color="blue",
                ps=10,
            )
        except Exception as e:
            logger.error(f"Error creating extracted keypoints visualization: {e}")

        # Build dataset for top-K matches table
        dataset_samples = []
        match_visualizations = {}

        for match in matches:
            rank = match["rank"]
            leopard_name = match["leopard_name"]
            wasserstein = match["wasserstein"]
            auc = match["auc"]
            num_matches = match["num_matches"]
            catalog_img_path = Path(match["filepath"])

            # Get location from catalog metadata
            catalog_id = match["catalog_id"]
            catalog_metadata = get_catalog_metadata_for_id(
                config.catalog_root, catalog_id
            )
            location = "unknown"
            if catalog_metadata:
                # Extract location from path: database/{location}/{individual}/...
                img_path_parts = Path(catalog_metadata["image_path"]).parts
                if len(img_path_parts) >= 3:
                    # Find 'database' in path and get next part
                    try:
                        db_idx = img_path_parts.index("database")
                        if db_idx + 1 < len(img_path_parts):
                            location = img_path_parts[db_idx + 1]
                    except ValueError:
                        pass

            # Confidence indicator (higher Wasserstein = better match)
            if wasserstein > 0.2:
                indicator = "✅"
            elif wasserstein > 0.1:
                indicator = "⚠️"
            else:
                indicator = "❓"

            # Create visualization for this match
            npz_path = pairwise_dir / f"rank_{rank:02d}_{match['catalog_id']}.npz"
            if npz_path.exists():
                try:
                    pairwise_data = np.load(npz_path)
                    match_viz = draw_matched_keypoints(
                        query_image_path=cropped_path,
                        catalog_image_path=catalog_img_path,
                        query_keypoints=pairwise_data["query_keypoints"],
                        catalog_keypoints=pairwise_data["catalog_keypoints"],
                        match_scores=pairwise_data["match_scores"],
                        max_matches=100,
                    )
                    match_visualizations[rank] = match_viz
                except Exception as e:
                    logger.error(f"Error creating visualization for rank {rank}: {e}")

            # Format for table (as list, not dict)
            dataset_samples.append(
                [
                    rank,
                    indicator,
                    leopard_name,
                    location,
                    f"{wasserstein:.4f}",
                    f"{auc:.4f}",
                    num_matches,
                ]
            )

        # Store match visualizations, enriched matches, filters, and temp_dir in global state
        LOADED_MODELS["current_match_visualizations"] = match_visualizations
        LOADED_MODELS["current_enriched_matches"] = matches
        LOADED_MODELS["current_filter_body_parts"] = filter_body_parts
        LOADED_MODELS["current_temp_dir"] = temp_dir

        return (
            result_text,
            seg_viz,
            cropped_image_pil,
            extracted_kpts_viz,
            dataset_samples,
            gr.update(visible=True),
        )

    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        return (
            f"❌ Error processing image: {str(e)}",
            None,
            None,
            None,
            [],
            gr.update(visible=False),
        )


def create_segmentation_viz(image_path, mask):
    """Create visualization of segmentation mask overlaid on image."""
    # Load original image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize mask to match image dimensions if needed
    if mask.shape[:2] != img_rgb.shape[:2]:
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (img_rgb.shape[1], img_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        mask_resized = mask

    # Create colored overlay
    overlay = img_rgb.copy()
    overlay[mask_resized > 0] = [255, 0, 0]  # Red for masked region

    # Blend
    alpha = 0.4
    blended = cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)

    return Image.fromarray(blended)


def on_match_selected(evt: gr.SelectData):
    """Handle selection of a match from the dataset table.

    Returns visualization, header, indicators, empty messages, and galleries organized by body part.
    """
    # evt.index is [row, col] for Dataframe, we want row
    if isinstance(evt.index, (list, tuple)):
        selected_row = evt.index[0]
    else:
        selected_row = evt.index

    selected_rank = selected_row + 1  # Ranks are 1-indexed

    # Get stored data from global state
    match_visualizations = LOADED_MODELS.get("current_match_visualizations", {})
    enriched_matches = LOADED_MODELS.get("current_enriched_matches", [])
    filter_body_parts = LOADED_MODELS.get("current_filter_body_parts")
    catalog_root = LOADED_MODELS.get("catalog_root")

    # Find the selected match
    selected_match = None
    for match in enriched_matches:
        if match["rank"] == selected_rank:
            selected_match = match
            break

    if not selected_match or selected_rank not in match_visualizations:
        # Return empty updates for all 17 outputs
        return (
            gr.update(visible=False),  # 1. visualization
            gr.update(value=""),  # 2. header
            gr.update(value=""),  # 3. head indicator
            gr.update(value=""),  # 4. left_flank indicator
            gr.update(value=""),  # 5. right_flank indicator
            gr.update(value=""),  # 6. tail indicator
            gr.update(value=""),  # 7. misc indicator
            gr.update(visible=False),  # 8. head empty message
            gr.update(visible=False),  # 9. left_flank empty message
            gr.update(visible=False),  # 10. right_flank empty message
            gr.update(visible=False),  # 11. tail empty message
            gr.update(visible=False),  # 12. misc empty message
            gr.update(value=[]),  # 13. head gallery
            gr.update(value=[]),  # 14. left_flank gallery
            gr.update(value=[]),  # 15. right_flank gallery
            gr.update(value=[]),  # 16. tail gallery
            gr.update(value=[]),  # 17. misc gallery
        )

    # Get visualization
    match_viz = match_visualizations[selected_rank]

    # Create dynamic header with leopard name
    leopard_name = selected_match["leopard_name"]
    header_text = f"## Reference Images for {leopard_name.title()}"

    # Load galleries organized by body part
    galleries = {}
    if catalog_root:
        try:
            # Extract location from match filepath
            location = None
            filepath = Path(selected_match["filepath"])
            parts = filepath.parts
            if "database" in parts:
                db_idx = parts.index("database")
                if db_idx + 1 < len(parts):
                    location = parts[db_idx + 1]

            galleries = load_matched_individual_gallery_by_body_part(
                catalog_root=catalog_root,
                leopard_name=leopard_name,
                location=location,
            )
        except Exception as e:
            logger.error(f"Error loading gallery for {leopard_name}: {e}")
            # Initialize empty galleries on error
            galleries = {
                "head": [],
                "left_flank": [],
                "right_flank": [],
                "tail": [],
                "misc": [],
            }

    # Create emoji indicators for filtered body parts
    def get_indicator(body_part: str) -> str:
        """Return ⭐ if body part was in filter, empty string otherwise."""
        if filter_body_parts and body_part in filter_body_parts:
            return "⭐"
        return ""

    # Helper to determine if empty message should be visible
    def is_empty(body_part: str) -> bool:
        """Return True if no images for this body part."""
        return len(galleries.get(body_part, [])) == 0

    return (
        gr.update(value=match_viz, visible=True),  # 1. visualization
        gr.update(value=header_text),  # 2. header
        gr.update(value=get_indicator("head")),  # 3. head indicator
        gr.update(value=get_indicator("left_flank")),  # 4. left_flank indicator
        gr.update(value=get_indicator("right_flank")),  # 5. right_flank indicator
        gr.update(value=get_indicator("tail")),  # 6. tail indicator
        gr.update(value=get_indicator("misc")),  # 7. misc indicator
        gr.update(visible=is_empty("head")),  # 8. head empty message
        gr.update(visible=is_empty("left_flank")),  # 9. left_flank empty message
        gr.update(visible=is_empty("right_flank")),  # 10. right_flank empty message
        gr.update(visible=is_empty("tail")),  # 11. tail empty message
        gr.update(visible=is_empty("misc")),  # 12. misc empty message
        gr.update(
            value=galleries.get("head", []), visible=not is_empty("head")
        ),  # 13. head gallery
        gr.update(
            value=galleries.get("left_flank", []), visible=not is_empty("left_flank")
        ),  # 14. left_flank gallery
        gr.update(
            value=galleries.get("right_flank", []), visible=not is_empty("right_flank")
        ),  # 15. right_flank gallery
        gr.update(
            value=galleries.get("tail", []), visible=not is_empty("tail")
        ),  # 16. tail gallery
        gr.update(
            value=galleries.get("misc", []), visible=not is_empty("misc")
        ),  # 17. misc gallery
    )


def load_matched_individual_gallery_by_body_part(
    catalog_root: Path,
    leopard_name: str,
    location: str | None = None,
) -> dict[str, list[tuple]]:
    """Load all images for a matched individual organized by body part.

    Args:
        catalog_root: Path to catalog root directory
        leopard_name: Name of the matched individual (e.g., "karindas")
        location: Geographic location (e.g., "naryn")

    Returns:
        Dict mapping body part to list of (PIL.Image, caption) tuples:
        {
            "head": [(img1, caption1), (img2, caption2), ...],
            "left_flank": [...],
            "right_flank": [...],
            "tail": [...],
            "misc": [...]
        }
    """
    # Initialize dict with all body parts
    galleries = {
        "head": [],
        "left_flank": [],
        "right_flank": [],
        "tail": [],
        "misc": [],
    }

    # Find metadata path: database/{location}/{individual}/metadata.yaml
    if location:
        metadata_path = (
            catalog_root / "database" / location / leopard_name / "metadata.yaml"
        )
    else:
        # Try to find the individual in any location
        metadata_path = None
        database_dir = catalog_root / "database"
        if database_dir.exists():
            for loc_dir in database_dir.iterdir():
                if loc_dir.is_dir():
                    potential_path = loc_dir / leopard_name / "metadata.yaml"
                    if potential_path.exists():
                        metadata_path = potential_path
                        break

    if not metadata_path or not metadata_path.exists():
        logger.warning(f"Metadata not found for {leopard_name}")
        return galleries

    try:
        metadata = load_leopard_metadata(metadata_path)

        # Load all images organized by body part
        for img_entry in metadata["reference_images"]:
            body_part = img_entry.get("body_part", "misc")

            # Normalize body_part to match our keys
            if body_part not in galleries:
                body_part = "misc"  # Default to misc if unknown

            # Load image
            img_path = catalog_root / "database" / img_entry["path"]

            try:
                img = Image.open(img_path)
                # Simple caption: just body part name
                caption = body_part
                galleries[body_part].append((img, caption))
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")

    except Exception as e:
        logger.error(f"Error loading metadata for {leopard_name}: {e}")

    return galleries


def cleanup_temp_files():
    """Clean up temporary files from previous run."""
    temp_dir = LOADED_MODELS.get("current_temp_dir")
    if temp_dir and temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temp directory: {e}")


def create_leopard_tab(leopard_metadata, config: AppConfig):
    """Create a tab for displaying a single leopard's images.

    Args:
        leopard_metadata: Metadata dictionary for the leopard individual
        config: Application configuration
    """
    # Support both 'leopard_name' and 'individual_name' keys
    leopard_name = leopard_metadata.get("leopard_name") or leopard_metadata.get(
        "individual_name"
    )
    location = leopard_metadata.get("location", "unknown")
    total_images = leopard_metadata["statistics"]["total_reference_images"]

    # Get body parts from statistics
    body_parts = leopard_metadata["statistics"].get(
        "body_parts_represented", leopard_metadata["statistics"].get("body_parts", [])
    )
    body_parts_str = ", ".join(body_parts) if body_parts else "N/A"

    with gr.Tab(f"{leopard_name}"):
        # Header with statistics
        gr.Markdown(
            f"### {leopard_name.title()}\n"
            f"**Location:** {location} | "
            f"**{total_images} images** | "
            f"**Body parts:** {body_parts_str}"
        )

        # Load all images with body_part captions
        gallery_data = []
        for img_entry in leopard_metadata["reference_images"]:
            img_path = config.catalog_root / "database" / img_entry["path"]
            body_part = img_entry.get("body_part", "unknown")
            try:
                img = Image.open(img_path)
                # Caption format: just body_part (location is already in tab)
                caption = body_part
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


def create_app(config: AppConfig):
    """Create and configure the Gradio application.

    Args:
        config: Application configuration
    """
    # Initialize models at startup
    initialize_models(config)

    # Load catalog data
    catalog_index, individuals_data = load_catalog_data(config)

    # Build example images list from examples directory
    example_images = (
        list(config.examples_dir.glob("*.jpg"))
        + list(config.examples_dir.glob("*.JPG"))
        + list(config.examples_dir.glob("*.png"))
    )

    # Create interface
    with gr.Blocks(title="Snow Leopard Identification", theme=gr.themes.Soft()) as app:
        gr.HTML("""
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="margin-bottom: 10px;">🐆 Snow Leopard Identification</h1>
                <p style="font-size: 16px; color: #666;">
                    Computer vision system for identifying individual snow leopards.
                </p>
            </div>
        """)

        # Main tabs
        with gr.Tabs():
            # Tab 1: Identify Snow Leopard
            with gr.Tab("🔍 Identify Snow Leopard"):
                gr.Markdown("""
Upload a snow leopard image or select an example to identify which individual it is.
The system will detect the leopard, extract distinctive features, and match against the catalog.
                """)

                with gr.Row():
                    # Left column: Input
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="pil",
                            label="Upload Snow Leopard Image",
                            sources=["upload", "clipboard"],
                        )

                        gr.Examples(
                            examples=[[str(img)] for img in example_images],
                            inputs=image_input,
                            label="Example Images",
                        )

                        # Feature extractor dropdown
                        available_extractors = get_available_extractors(
                            config.catalog_root
                        )
                        extractor_dropdown = gr.Dropdown(
                            choices=available_extractors,
                            value=available_extractors[0]
                            if available_extractors
                            else "sift",
                            label="Feature Extractor",
                            info=f"Select extractor (available in catalog: {', '.join(available_extractors)})",
                        )

                        # Location filter dropdown
                        available_locations = get_available_locations(
                            config.catalog_root
                        )
                        location_filter = gr.Dropdown(
                            choices=available_locations,
                            value=["all"],
                            multiselect=True,
                            label="Filter by Location",
                            info="Select locations to search (default: all locations)",
                        )

                        # Body part filter dropdown
                        available_body_parts = get_available_body_parts(
                            config.catalog_root
                        )
                        body_part_filter = gr.Dropdown(
                            choices=available_body_parts,
                            value=["all"],
                            multiselect=True,
                            label="Filter by Body Part",
                            info="Select body parts to match (default: all body parts)",
                        )

                        submit_btn = gr.Button(
                            value="🔍 Identify Snow Leopard",
                            variant="primary",
                            size="lg",
                        )

                    # Right column: Results
                    with gr.Column(scale=4):
                        # Top-1 prediction
                        result_text = gr.Markdown("")

                        # Tabs for different result views
                        with gr.Tabs():
                            with gr.Tab("Model Internals"):
                                gr.Markdown("""
View the internal processing steps: segmentation mask, cropped leopard, and extracted keypoints.
                                """)
                                with gr.Row():
                                    seg_viz = gr.Image(
                                        label="Segmentation Overlay",
                                        type="pil",
                                    )
                                    cropped_image = gr.Image(
                                        label="Extracted Snow Leopard",
                                        type="pil",
                                    )
                                    extracted_kpts_viz = gr.Image(
                                        label="Extracted Keypoints",
                                        type="pil",
                                    )

                            with gr.Tab("Top Matches"):
                                gr.Markdown("""
Click a row to view detailed feature matching visualization and all reference images for that leopard.
**Higher Wasserstein distance = better match** (typical range: 0.1-0.3)
                                """)

                                matches_dataset = gr.Dataframe(
                                    headers=[
                                        "Rank",
                                        "Confidence",
                                        "Leopard Name",
                                        "Location",
                                        "Wasserstein",
                                        "AUC",
                                        "Matches",
                                    ],
                                    label="Top 5 Matches",
                                    interactive=False,
                                    wrap=True,
                                    row_count=(5, "fixed"),
                                    col_count=(7, "fixed"),
                                )

                                selected_match_viz = gr.Image(
                                    label="Selected Match Visualization (Feature Keypoints)",
                                    type="pil",
                                    visible=False,
                                )

                                # Dynamic header showing matched leopard name
                                selected_match_header = gr.Markdown("", visible=True)

                                # Create tabs for each body part
                                with gr.Tabs():
                                    with gr.Tab("🗣️ Head"):
                                        head_indicator = gr.Markdown("")
                                        head_empty_message = gr.Markdown(
                                            value='<div style="text-align: center; padding: 60px 20px; color: #888;">'
                                            '<p style="font-size: 16px;">No reference images available for this body part</p>'
                                            "</div>",
                                            visible=False,
                                        )
                                        gallery_head = gr.Gallery(
                                            columns=6,
                                            height=400,
                                            object_fit="scale-down",
                                            allow_preview=True,
                                        )

                                    with gr.Tab("⬅️ Left Flank"):
                                        left_flank_indicator = gr.Markdown("")
                                        left_flank_empty_message = gr.Markdown(
                                            value='<div style="text-align: center; padding: 60px 20px; color: #888;">'
                                            '<p style="font-size: 16px;">No reference images available for this body part</p>'
                                            "</div>",
                                            visible=False,
                                        )
                                        gallery_left_flank = gr.Gallery(
                                            columns=6,
                                            height=400,
                                            object_fit="scale-down",
                                            allow_preview=True,
                                        )

                                    with gr.Tab("➡️ Right Flank"):
                                        right_flank_indicator = gr.Markdown("")
                                        right_flank_empty_message = gr.Markdown(
                                            value='<div style="text-align: center; padding: 60px 20px; color: #888;">'
                                            '<p style="font-size: 16px;">No reference images available for this body part</p>'
                                            "</div>",
                                            visible=False,
                                        )
                                        gallery_right_flank = gr.Gallery(
                                            columns=6,
                                            height=400,
                                            object_fit="scale-down",
                                            allow_preview=True,
                                        )

                                    with gr.Tab("🪶 Tail"):
                                        tail_indicator = gr.Markdown("")
                                        tail_empty_message = gr.Markdown(
                                            value='<div style="text-align: center; padding: 60px 20px; color: #888;">'
                                            '<p style="font-size: 16px;">No reference images available for this body part</p>'
                                            "</div>",
                                            visible=False,
                                        )
                                        gallery_tail = gr.Gallery(
                                            columns=6,
                                            height=400,
                                            object_fit="scale-down",
                                            allow_preview=True,
                                        )

                                    with gr.Tab("📋 Other"):
                                        misc_indicator = gr.Markdown("")
                                        misc_empty_message = gr.Markdown(
                                            value='<div style="text-align: center; padding: 60px 20px; color: #888;">'
                                            '<p style="font-size: 16px;">No reference images available for this body part</p>'
                                            "</div>",
                                            visible=False,
                                        )
                                        gallery_misc = gr.Gallery(
                                            columns=6,
                                            height=400,
                                            object_fit="scale-down",
                                            allow_preview=True,
                                        )

                # Connect submit button
                submit_btn.click(
                    fn=lambda img, ext, locs, parts: run_identification(
                        image=img,
                        extractor=ext,
                        selected_locations=locs,
                        selected_body_parts=parts,
                        config=config,
                    ),
                    inputs=[
                        image_input,
                        extractor_dropdown,
                        location_filter,
                        body_part_filter,
                    ],
                    outputs=[
                        result_text,
                        seg_viz,
                        cropped_image,
                        extracted_kpts_viz,
                        matches_dataset,
                        selected_match_viz,
                    ],
                )

                # Connect dataset selection
                matches_dataset.select(
                    fn=on_match_selected,
                    outputs=[
                        selected_match_viz,
                        selected_match_header,
                        head_indicator,
                        left_flank_indicator,
                        right_flank_indicator,
                        tail_indicator,
                        misc_indicator,
                        head_empty_message,
                        left_flank_empty_message,
                        right_flank_empty_message,
                        tail_empty_message,
                        misc_empty_message,
                        gallery_head,
                        gallery_left_flank,
                        gallery_right_flank,
                        gallery_tail,
                        gallery_misc,
                    ],
                )

            # Tab 2: Explore Catalog
            with gr.Tab("📚 Explore Catalog"):
                gr.Markdown(
                    """
                    ## Snow Leopard Catalog Browser

                    Browse the reference catalog of known snow leopard individuals.
                    Each individual has multiple reference images from different body parts and locations.
                    """
                )

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
                for individual_data in individuals_data:
                    location = individual_data.get("location", "unknown")
                    if location not in individuals_by_location:
                        individuals_by_location[location] = []
                    individuals_by_location[location].append(individual_data)

                # Create tabs for each location
                with gr.Tabs():
                    for location in sorted(individuals_by_location.keys()):
                        with gr.Tab(f"📍 {location.title()}"):
                            # Create subtabs for each individual in this location
                            with gr.Tabs():
                                for leopard_data in individuals_by_location[location]:
                                    create_leopard_tab(
                                        leopard_metadata=leopard_data, config=config
                                    )

        # Cleanup on app close
        app.unload(cleanup_temp_files)

        # Load first example image on startup
        def load_first_example():
            """Load the first example image when the app starts."""
            if example_images:
                try:
                    first_image = Image.open(example_images[0])
                    return first_image
                except Exception as e:
                    logger.error(f"Error loading first example image: {e}")
                    return None
            return None

        app.load(fn=load_first_example, outputs=[image_input])

    return app


if __name__ == "__main__":
    config = parse_args()
    app = create_app(config)
    app.launch(server_name="127.0.0.1", server_port=config.port, share=config.share)
