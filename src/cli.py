"""Command line interface for single-image scene perception."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import cv2

from .association import attach_depth_to_detections
from .depth import DepthEstimator
from .detection import ObjectDetector
from .schemas import DetectionWithDepth
from .visualization import (
    compose_side_by_side,
    draw_detections,
    render_depth_heatmap,
    save_image,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run single-image scene perception with YOLO11 and MiDaS.",
    )
    parser.add_argument("--image", required=True, help="Path to one input image.")
    parser.add_argument(
        "--output",
        default="artifacts",
        help="Output directory for images and JSON results.",
    )
    parser.add_argument("--detector-model", default="yolo11n.pt", help="YOLO model name or path.")
    parser.add_argument("--depth-model", default="DPT_Hybrid", help="MiDaS model type.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Minimum YOLO detection confidence.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        output_paths = run_pipeline(
            image_path=Path(args.image),
            output_dir=Path(args.output),
            detector_model=args.detector_model,
            depth_model=args.depth_model,
            confidence_threshold=args.confidence_threshold,
        )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    for label, path in output_paths.items():
        print(f"Saved {label}: {path}")
    return 0


def run_pipeline(
    image_path: Path,
    output_dir: Path,
    detector_model: str,
    depth_model: str,
    confidence_threshold: float,
) -> dict[str, Path]:
    image = load_image(image_path)
    images_dir, json_dir = create_output_dirs(output_dir)

    detector = ObjectDetector(
        model_name=detector_model,
        confidence_threshold=confidence_threshold,
    )
    detections = detector.predict(str(image_path))

    depth_estimator = DepthEstimator(model_type=depth_model)
    depth_map = depth_estimator.predict(str(image_path))

    detections_with_depth = attach_depth_to_detections(detections, depth_map)
    detection_image = draw_detections(image, detections_with_depth)
    depth_heatmap = render_depth_heatmap(depth_map)
    combined_image = compose_side_by_side(image, detection_image, depth_heatmap)

    stem = image_path.stem
    detection_path = images_dir / f"{stem}_detections.jpg"
    depth_path = images_dir / f"{stem}_depth_heatmap.jpg"
    combined_path = images_dir / f"{stem}_combined.jpg"
    json_path = json_dir / f"{stem}_result.json"

    save_image(detection_path, detection_image)
    save_image(depth_path, depth_heatmap)
    save_image(combined_path, combined_image)
    save_json_result(json_path, image_path, detections_with_depth)

    return {
        "detection overlay": detection_path,
        "depth heatmap": depth_path,
        "combined image": combined_path,
        "JSON result": json_path,
    }


def load_image(image_path: Path) -> cv2.typing.MatLike:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Input path is not a file: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read input image: {image_path}")
    return image


def create_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    images_dir = output_dir / "images"
    json_dir = output_dir / "json"
    images_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, json_dir


def save_json_result(
    path: Path,
    image_path: Path,
    detections_with_depth: list[DetectionWithDepth],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": str(image_path),
        "detections": [detection.to_dict() for detection in detections_with_depth],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
