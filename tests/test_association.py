from __future__ import annotations

import numpy as np

from src.association import associate_depth, median_depth_for_box
from src.schemas import BBox, Detection


def test_median_depth_for_box() -> None:
    depth_map = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )

    assert median_depth_for_box(depth_map, (0, 0, 2, 2)) == 3.0


def test_associate_depth() -> None:
    depth_map = np.ones((4, 4), dtype=float) * 2.5
    detections = [
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=BBox(x1=1, y1=1, x2=3, y2=3),
        )
    ]

    objects = associate_depth(detections, depth_map)

    assert len(objects) == 1
    assert objects[0].class_name == "person"
    assert objects[0].relative_depth_score == 2.5
