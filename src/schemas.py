"""Pure Python data schemas shared across the perception pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BBox:
    """Bounding box in xyxy format."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x1 < 0 or self.y1 < 0:
            raise ValueError(f"BBox coordinates must be non-negative: {self.as_xyxy()}")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"Invalid BBox coordinates: {self.as_xyxy()}")

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class Detection:
    """Object detection result without depth information."""

    class_id: int
    class_name: str
    confidence: float
    bbox: BBox

    def __post_init__(self) -> None:
        if self.class_id < 0:
            raise ValueError(f"class_id must be non-negative: {self.class_id}")
        if not self.class_name:
            raise ValueError("class_name must not be empty")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"confidence must be in [0, 1]: {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": list(self.bbox.as_xyxy()),
        }


@dataclass(frozen=True)
class DetectionWithDepth(Detection):
    """Detection with an approximate relative depth score."""

    relative_depth_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["relative_depth_score"] = self.relative_depth_score
        return data
