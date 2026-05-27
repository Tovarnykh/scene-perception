"""YOLO11 object detection wrapper."""

from __future__ import annotations

from typing import Any

from .schemas import BBox, Detection


class ObjectDetector:
    """Thin wrapper around Ultralytics YOLO11.

    Example:
        detector = ObjectDetector("yolo11n.pt", confidence_threshold=0.25)
        detections = detector.predict("data/samples/street.jpg")
        print([detection.to_dict() for detection in detections])
    """

    def __init__(self, model_name: str = "yolo11n.pt", confidence_threshold: float = 0.25) -> None:
        if confidence_threshold < 0.0 or confidence_threshold > 1.0:
            raise ValueError(f"confidence_threshold must be in [0, 1]: {confidence_threshold}")

        from ultralytics import YOLO

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(model_name)

    def predict(self, image_path: str) -> list[Detection]:
        """Run YOLO11 on one image and return Detection dataclasses."""
        results = self.model.predict(
            source=image_path,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            detections.extend(self._detections_from_result(result))
        return detections

    def _detections_from_result(self, result: Any) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        names = getattr(result, "names", None) or getattr(self.model, "names", {})
        xyxy_values = boxes.xyxy.tolist()
        confidence_values = boxes.conf.tolist()
        class_values = boxes.cls.tolist()

        detections: list[Detection] = []
        for xyxy, confidence, class_id_value in zip(
            xyxy_values,
            confidence_values,
            class_values,
            strict=True,
        ):
            confidence_float = float(confidence)
            if confidence_float < self.confidence_threshold:
                continue

            class_id = int(class_id_value)
            class_name = _class_name_from_names(names, class_id)
            bbox = _bbox_from_xyxy(xyxy)
            if bbox is None:
                continue

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence_float,
                    bbox=bbox,
                )
            )

        return detections


def _bbox_from_xyxy(xyxy: list[float]) -> BBox | None:
    x1, y1, x2, y2 = (int(round(value)) for value in xyxy)
    try:
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2)
    except ValueError:
        return None


def _class_name_from_names(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def detect_objects(image_path: str) -> list[Detection]:
    """Smoke helper using the default YOLO11 nano model."""
    return ObjectDetector().predict(image_path)
