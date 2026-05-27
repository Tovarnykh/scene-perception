"""Visualization helpers for perception outputs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .schemas import DetectionWithDepth

BoxColor = tuple[int, int, int]


def draw_detections(
    image: np.ndarray,
    detections_with_depth: list[DetectionWithDepth],
) -> np.ndarray:
    """Draw detections and approximate relative depth scores on an image."""
    output = image.copy()
    for detection in detections_with_depth:
        _draw_detection(output, detection)
    return output


def render_depth_heatmap(depth_map: np.ndarray) -> np.ndarray:
    """Render a relative depth map as a color heatmap."""
    normalized = _normalize_depth_map(depth_map)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def compose_side_by_side(
    original_image: np.ndarray,
    detection_image: np.ndarray,
    depth_heatmap: np.ndarray,
) -> np.ndarray:
    """Compose original, annotated, and depth heatmap images side by side."""
    height, width = original_image.shape[:2]
    detection_resized = _resize_to(detection_image, width, height)
    heatmap_resized = _resize_to(depth_heatmap, width, height)
    return np.hstack([original_image, detection_resized, heatmap_resized])


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Save an image array to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Failed to save image: {output_path}")


def format_object_label(detection: DetectionWithDepth) -> str:
    """Format a compact label for drawing on images."""
    depth = detection.relative_depth_score
    depth_text = "n/a" if depth is None else f"{depth:.2f}"
    return f"{detection.class_name} {detection.confidence:.2f} depth {depth_text}"


def _draw_detection(
    image: np.ndarray,
    detection: DetectionWithDepth,
    color: BoxColor = (0, 255, 0),
) -> None:
    x1, y1, x2, y2 = detection.bbox.as_xyxy()
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
    _draw_label(image, format_object_label(detection), x1, y1, color)


def _draw_label(image: np.ndarray, text: str, x: int, y: int, color: BoxColor) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    label_y = max(y, text_height + baseline + 4)

    top_left = (x, label_y - text_height - baseline - 4)
    bottom_right = (x + text_width + 6, label_y)
    cv2.rectangle(image, top_left, bottom_right, color, thickness=-1)
    cv2.putText(
        image,
        text,
        (x + 3, label_y - baseline - 2),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def _normalize_depth_map(depth_map: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(depth_map)
    if not finite_mask.any():
        return np.zeros(depth_map.shape, dtype=np.uint8)

    finite_values = depth_map[finite_mask]
    min_value = float(finite_values.min())
    max_value = float(finite_values.max())
    if max_value == min_value:
        return np.zeros(depth_map.shape, dtype=np.uint8)

    normalized = np.zeros(depth_map.shape, dtype=np.float32)
    normalized[finite_mask] = (depth_map[finite_mask] - min_value) / (max_value - min_value)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def _resize_to(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
