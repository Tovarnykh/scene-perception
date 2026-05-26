from __future__ import annotations

import pytest

from src.schemas import Detection, SceneObject


def test_detection_accepts_valid_bbox_and_confidence() -> None:
    detection = Detection(label="car", confidence=0.75, bbox=(0, 1, 10, 12))

    assert detection.label == "car"


def test_detection_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        Detection(label="car", confidence=1.5, bbox=(0, 0, 10, 10))


def test_scene_object_serializes_to_dict() -> None:
    scene_object = SceneObject(label="chair", confidence=0.8, bbox=(1, 2, 3, 4), depth=5.5)

    assert scene_object.to_dict() == {
        "label": "chair",
        "confidence": 0.8,
        "bbox": (1, 2, 3, 4),
        "depth": 5.5,
    }
