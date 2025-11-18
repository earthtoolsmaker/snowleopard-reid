"""Pipeline stages for snow leopard re-identification.

This module contains all pipeline stages that process query images through
segmentation, feature extraction, and matching.
"""

from .feature_extraction import run_feature_extraction_stage
from .mask_selection import run_mask_selection_stage, select_best_mask
from .matching import run_matching_stage
from .preprocess import run_preprocess_stage
from .segmentation import run_segmentation_stage

__all__ = [
    "run_segmentation_stage",
    "run_mask_selection_stage",
    "select_best_mask",
    "run_preprocess_stage",
    "run_feature_extraction_stage",
    "run_matching_stage",
]
