"""Depth estimation integration boundary."""

from __future__ import annotations

import numpy as np


def estimate_depth(image_path: str) -> np.ndarray:
    """Estimate a depth map for an image."""
    raise NotImplementedError(f"MiDaS depth estimation is not implemented yet: {image_path}")
