"""Object detection integration boundary."""

from __future__ import annotations

from .schemas import Detection


def detect_objects(image_path: str) -> list[Detection]:
    """Run object detection for an image.

    The implementation is intentionally deferred so tests do not require model
    weights or GPU access.
    """
    raise NotImplementedError(f"YOLO11 detection is not implemented yet: {image_path}")
