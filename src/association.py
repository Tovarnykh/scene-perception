"""Association logic between detections and depth maps."""

from __future__ import annotations

import numpy as np

from .schemas import BBox, Detection, DetectionWithDepth


def median_depth_for_box(depth_map: np.ndarray, box: BBox | tuple[int, int, int, int]) -> float:
    """Return median depth inside an inclusive-exclusive bounding box."""
    x1, y1, x2, y2 = box.as_xyxy() if isinstance(box, BBox) else box
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid box: {box}")
    crop = depth_map[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Box is outside depth map or empty: {box}")
    return float(np.median(crop))


def associate_depth(detections: list[Detection], depth_map: np.ndarray) -> list[DetectionWithDepth]:
    """Attach approximate relative depth scores to detections."""
    return [
        DetectionWithDepth(
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox=detection.bbox,
            relative_depth_score=median_depth_for_box(depth_map, detection.bbox),
        )
        for detection in detections
    ]
