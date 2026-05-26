"""Visualization helpers for perception outputs."""

from __future__ import annotations

from .schemas import SceneObject


def format_object_label(scene_object: SceneObject) -> str:
    """Format a compact label for drawing on images."""
    return f"{scene_object.label} {scene_object.confidence:.2f} depth={scene_object.depth:.2f}"
