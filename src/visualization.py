"""Visualization helpers for perception outputs."""

from __future__ import annotations

from .schemas import DetectionWithDepth


def format_object_label(detection: DetectionWithDepth) -> str:
    """Format a compact label for drawing on images."""
    depth = detection.relative_depth_score
    depth_text = "n/a" if depth is None else f"{depth:.2f}"
    return f"{detection.class_name} {detection.confidence:.2f} relative_depth={depth_text}"
