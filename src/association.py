"""Association logic between detections and depth maps."""

from __future__ import annotations

import numpy as np

from .schemas import Detection, SceneObject


def median_depth_for_box(depth_map: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Return median depth inside an inclusive-exclusive bounding box."""
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid box: {box}")
    crop = depth_map[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Box is outside depth map or empty: {box}")
    return float(np.median(crop))


def associate_depth(detections: list[Detection], depth_map: np.ndarray) -> list[SceneObject]:
    """Attach median depth values to detections."""
    return [
        SceneObject(
            label=detection.label,
            confidence=detection.confidence,
            bbox=detection.bbox,
            depth=median_depth_for_box(depth_map, detection.bbox),
        )
        for detection in detections
    ]
