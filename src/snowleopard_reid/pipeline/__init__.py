"""Pipeline module for snow leopard re-identification.

This module provides the complete end-to-end pipeline for identifying individual
snow leopards from query images.
"""

from .stages import (
    run_feature_extraction_stage,
    run_mask_selection_stage,
    run_matching_stage,
    run_preprocess_stage,
    run_segmentation_stage,
    select_best_mask,
)

__all__ = [
    "run_segmentation_stage",
    "run_mask_selection_stage",
    "run_preprocess_stage",
    "run_feature_extraction_stage",
    "run_matching_stage",
    "select_best_mask",
]
