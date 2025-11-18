"""Catalog module for snow leopard re-identification.

This module provides utilities for loading and managing the snow leopard catalog,
including individual metadata and feature data.
"""

from .loader import (
    get_all_catalog_features,
    get_available_body_parts,
    get_available_locations,
    get_catalog_metadata_for_id,
    get_filtered_catalog_features,
    load_catalog_index,
    load_leopard_metadata,
)

__all__ = [
    "load_catalog_index",
    "load_leopard_metadata",
    "get_all_catalog_features",
    "get_filtered_catalog_features",
    "get_available_locations",
    "get_available_body_parts",
    "get_catalog_metadata_for_id",
]
