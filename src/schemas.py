"""Typed data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: BBox

    def __post_init__(self) -> None:
        validate_bbox(self.bbox)
        validate_confidence(self.confidence)


@dataclass(frozen=True)
class SceneObject:
    label: str
    confidence: float
    bbox: BBox
    depth: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_bbox(bbox: BBox) -> None:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox coordinates: {bbox}")


def validate_confidence(confidence: float) -> None:
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"Confidence must be in [0, 1]: {confidence}")
