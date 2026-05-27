from __future__ import annotations

import pytest

from src.schemas import BBox, Detection, DetectionWithDepth


def test_bbox_as_xyxy() -> None:
    bbox = BBox(x1=0, y1=1, x2=10, y2=12)

    assert bbox.as_xyxy() == (0, 1, 10, 12)


def test_bbox_rejects_invalid_coordinates() -> None:
    with pytest.raises(ValueError):
        BBox(x1=10, y1=1, x2=10, y2=12)


def test_detection_accepts_valid_fields() -> None:
    detection = Detection(
        class_id=2,
        class_name="car",
        confidence=0.75,
        bbox=BBox(x1=0, y1=1, x2=10, y2=12),
    )

    assert detection.class_name == "car"


def test_detection_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        Detection(
            class_id=2,
            class_name="car",
            confidence=1.5,
            bbox=BBox(x1=0, y1=0, x2=10, y2=10),
        )


def test_detection_serializes_to_json_friendly_dict() -> None:
    detection = Detection(
        class_id=56,
        class_name="chair",
        confidence=0.8,
        bbox=BBox(x1=1, y1=2, x2=3, y2=4),
    )

    assert detection.to_dict() == {
        "class_id": 56,
        "class_name": "chair",
        "confidence": 0.8,
        "bbox": [1, 2, 3, 4],
    }


def test_detection_with_depth_serializes_to_json_friendly_dict() -> None:
    detection = DetectionWithDepth(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=BBox(x1=5, y1=6, x2=15, y2=16),
        relative_depth_score=0.42,
    )

    assert detection.to_dict() == {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.9,
        "bbox": [5, 6, 15, 16],
        "relative_depth_score": 0.42,
    }
