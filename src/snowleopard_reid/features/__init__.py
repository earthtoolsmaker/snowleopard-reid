"""Features module for snow leopard re-identification.

This module provides utilities for extracting, loading, and saving features
from snow leopard images using various feature extractors (SIFT, etc.).
"""

from .extraction import (
    extract_features,
    extract_sift_features,
    get_num_keypoints,
    load_features,
    save_features,
)

__all__ = [
    "extract_features",
    "extract_sift_features",
    "load_features",
    "save_features",
    "get_num_keypoints",
]
