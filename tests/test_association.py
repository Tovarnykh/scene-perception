from __future__ import annotations

import numpy as np

from src.association import (
    attach_depth_to_detections,
    clip_bbox_to_image,
    compute_relative_depth_score,
)
from src.schemas import BBox, Detection


def test_compute_relative_depth_score_normal_bbox() -> None:
    depth_map = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )

    assert compute_relative_depth_score(depth_map, BBox(x1=0, y1=0, x2=2, y2=2)) == 3.0


def test_clip_bbox_to_image_at_image_edge() -> None:
    assert clip_bbox_to_image(BBox(x1=2, y1=1, x2=5, y2=4), 4, 3) == (2, 1, 4, 3)


def test_compute_relative_depth_score_handles_bbox_at_image_edge() -> None:
    depth_map = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    )

    assert compute_relative_depth_score(depth_map, BBox(x1=2, y1=1, x2=6, y2=5)) == 9.5


def test_compute_relative_depth_score_returns_none_for_bbox_outside_image() -> None:
    depth_map = np.ones((3, 3), dtype=float)

    assert compute_relative_depth_score(depth_map, BBox(x1=4, y1=4, x2=5, y2=5)) is None


def test_compute_relative_depth_score_ignores_nan_and_infinite_values() -> None:
    depth_map = np.array(
        [
            [1.0, np.nan, np.inf],
            [4.0, 5.0, -np.inf],
            [7.0, 8.0, 9.0],
        ]
    )

    assert compute_relative_depth_score(depth_map, BBox(x1=0, y1=0, x2=3, y2=2)) == 4.0


def test_attach_depth_to_detections_handles_empty_detections() -> None:
    depth_map = np.ones((3, 3), dtype=float)

    assert attach_depth_to_detections([], depth_map) == []


def test_attach_depth_to_detections_returns_deterministic_output_dict() -> None:
    depth_map = np.ones((4, 4), dtype=float) * 2.5
    detections = [
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=BBox(x1=1, y1=1, x2=3, y2=3),
        )
    ]

    objects = attach_depth_to_detections(detections, depth_map)

    assert [item.to_dict() for item in objects] == [
        {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.9,
            "bbox": [1, 1, 3, 3],
            "relative_depth_score": 2.5,
        }
    ]
