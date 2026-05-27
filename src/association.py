"""Pure NumPy association logic between detections and relative depth maps."""

from __future__ import annotations

import numpy as np

from .schemas import BBox, Detection, DetectionWithDepth

RawBBox = BBox | tuple[int, int, int, int]


def _as_xyxy(bbox: RawBBox) -> tuple[int, int, int, int]:
    return bbox.as_xyxy() if isinstance(bbox, BBox) else bbox


def clip_bbox_to_image(
    bbox: RawBBox,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """Clip an xyxy bbox to image bounds, returning None if it has no area."""
    if image_width <= 0 or image_height <= 0:
        return None

    x1, y1, x2, y2 = _as_xyxy(bbox)
    clipped = (
        max(0, min(image_width, x1)),
        max(0, min(image_height, y1)),
        max(0, min(image_width, x2)),
        max(0, min(image_height, y2)),
    )
    clipped_x1, clipped_y1, clipped_x2, clipped_y2 = clipped
    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        return None
    return clipped


def compute_relative_depth_score(depth_map: np.ndarray, bbox: RawBBox) -> float | None:
    """Return median finite relative depth inside a clipped bbox."""
    if depth_map.ndim != 2:
        raise ValueError(f"depth_map must be 2D, got shape {depth_map.shape}")

    image_height, image_width = depth_map.shape
    clipped_bbox = clip_bbox_to_image(bbox, image_width=image_width, image_height=image_height)
    if clipped_bbox is None:
        return None

    x1, y1, x2, y2 = clipped_bbox
    crop = depth_map[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    finite_values = crop[np.isfinite(crop)]
    if finite_values.size == 0:
        return None

    return float(np.median(finite_values))


def attach_depth_to_detections(
    detections: list[Detection],
    depth_map: np.ndarray,
) -> list[DetectionWithDepth]:
    """Attach approximate relative depth scores to detections."""
    return [
        DetectionWithDepth(
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox=detection.bbox,
            relative_depth_score=compute_relative_depth_score(depth_map, detection.bbox),
        )
        for detection in detections
    ]


def median_depth_for_box(depth_map: np.ndarray, box: RawBBox) -> float | None:
    """Backward-compatible alias for relative depth scoring."""
    return compute_relative_depth_score(depth_map, box)


def associate_depth(detections: list[Detection], depth_map: np.ndarray) -> list[DetectionWithDepth]:
    """Backward-compatible alias for attaching relative depth scores."""
    return attach_depth_to_detections(detections, depth_map)
