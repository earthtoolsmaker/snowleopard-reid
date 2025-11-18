"""Visualization utilities for snow leopard re-identification.

This module provides functions for visualizing keypoints, matches, and other
pipeline outputs for debugging and presentation.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def draw_keypoints_overlay(
    image_path: Path | str,
    keypoints: np.ndarray,
    max_keypoints: int = 500,
    color: str = "blue",
    ps: int = 10,
) -> Image.Image:
    """Draw keypoints overlaid on an image.

    Args:
        image_path: Path to image file
        keypoints: Keypoints array of shape [N, 2] with (x, y) coordinates
        max_keypoints: Maximum number of keypoints to display
        color: Color name ('blue', 'red', 'green', etc.)
        ps: Point size for keypoints

    Returns:
        PIL Image with keypoints drawn
    """
    # Load image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Color mapping
    color_map = {
        "blue": (0, 0, 255),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
    }
    color_rgb = color_map.get(color.lower(), (0, 0, 255))

    # Draw keypoints (limit to max_keypoints)
    n_keypoints = min(len(keypoints), max_keypoints)
    for i in range(n_keypoints):
        x, y = keypoints[i]
        cv2.circle(img_rgb, (int(x), int(y)), ps // 2, color_rgb, -1)

    return Image.fromarray(img_rgb)


def draw_matched_keypoints(
    query_image_path: Path | str,
    catalog_image_path: Path | str,
    query_keypoints: np.ndarray,
    catalog_keypoints: np.ndarray,
    match_scores: np.ndarray,
    max_matches: int = 100,
) -> Image.Image:
    """Draw matched keypoints side-by-side with connecting lines.

    Args:
        query_image_path: Path to query image
        catalog_image_path: Path to catalog image
        query_keypoints: Query keypoints [N, 2]
        catalog_keypoints: Catalog keypoints [N, 2]
        match_scores: Match confidence scores [N]
        max_matches: Maximum number of matches to display

    Returns:
        PIL Image with side-by-side images and match lines
    """
    # Load images
    query_img = cv2.imread(str(query_image_path))
    catalog_img = cv2.imread(str(catalog_image_path))

    # Convert to RGB
    query_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    catalog_rgb = cv2.cvtColor(catalog_img, cv2.COLOR_BGR2RGB)

    # Resize images to same height for side-by-side display
    max_height = 800
    query_h, query_w = query_rgb.shape[:2]
    catalog_h, catalog_w = catalog_rgb.shape[:2]

    # Calculate scaling factors
    if query_h > max_height or catalog_h > max_height:
        query_scale = max_height / query_h
        catalog_scale = max_height / catalog_h
    else:
        query_scale = 1.0
        catalog_scale = 1.0

    # Resize images
    new_query_h = int(query_h * query_scale)
    new_query_w = int(query_w * query_scale)
    new_catalog_h = int(catalog_h * catalog_scale)
    new_catalog_w = int(catalog_w * catalog_scale)

    query_resized = cv2.resize(query_rgb, (new_query_w, new_query_h))
    catalog_resized = cv2.resize(catalog_rgb, (new_catalog_w, new_catalog_h))

    # Scale keypoints
    query_kpts_scaled = query_keypoints * query_scale
    catalog_kpts_scaled = catalog_keypoints * catalog_scale

    # Create side-by-side canvas
    combined_h = max(new_query_h, new_catalog_h)
    combined_w = new_query_w + new_catalog_w
    canvas = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)

    # Place images on canvas
    canvas[:new_query_h, :new_query_w] = query_resized
    canvas[:new_catalog_h, new_query_w : new_query_w + new_catalog_w] = catalog_resized

    # Offset catalog keypoints to account for horizontal placement
    catalog_kpts_offset = catalog_kpts_scaled.copy()
    catalog_kpts_offset[:, 0] += new_query_w

    # Draw matches (limit to max_matches)
    n_matches = min(len(query_kpts_scaled), max_matches)

    # Sort by match scores (highest confidence first)
    if len(match_scores) > 0:
        sorted_indices = np.argsort(match_scores)[::-1][:n_matches]
    else:
        sorted_indices = np.arange(n_matches)

    # Draw lines and keypoints
    for idx in sorted_indices:
        query_pt = tuple(query_kpts_scaled[idx].astype(int))
        catalog_pt = tuple(catalog_kpts_offset[idx].astype(int))

        # Color based on match score (green = high, yellow = medium, red = low)
        score = match_scores[idx] if len(match_scores) > 0 else 0.5
        if score > 0.8:
            color = (0, 255, 0)  # Green
        elif score > 0.5:
            color = (255, 255, 0)  # Yellow
        else:
            color = (255, 0, 0)  # Red

        # Draw line
        cv2.line(canvas, query_pt, catalog_pt, color, 1)

        # Draw keypoints
        cv2.circle(canvas, query_pt, 5, (255, 0, 0), -1)
        cv2.circle(canvas, catalog_pt, 5, (0, 0, 255), -1)

    return Image.fromarray(canvas)
