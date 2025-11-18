"""Utility functions for snow leopard re-identification.

This module provides common utilities used across the project.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def get_device(device: str | None = None, verbose: bool = True) -> str:
    """Get the device to use for computation.

    Auto-detects GPU if available, or uses CPU as fallback.
    Optionally allows manual override.

    Args:
        device: Manual device override ('cpu', 'cuda', or None for auto-detect)
        verbose: Whether to log device information

    Returns:
        Device string ('cuda' or 'cpu')

    Examples:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device('cpu')  # Force CPU
        >>> device = get_device('cuda')  # Force CUDA (will fail if not available)
    """
    if device is None:
        # Auto-detect
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if verbose:
            if device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                logger.info("Using CPU (no GPU available)")
    else:
        # Manual override
        device = device.lower()
        if device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid device: {device}. Must be 'cpu' or 'cuda'")

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but CUDA is not available. "
                "Install CUDA-enabled PyTorch or use device='cpu'"
            )

        if verbose:
            logger.info(f"Using device: {device}")

    return device
