"""MiDaS relative depth estimation wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class DepthEstimator:
    """Thin wrapper around MiDaS monocular relative depth estimation.

    Example:
        estimator = DepthEstimator("DPT_Hybrid")
        relative_depth = estimator.predict("data/samples/street.jpg")
        print(relative_depth.shape)
    """

    def __init__(self, model_type: str = "DPT_Hybrid") -> None:
        import torch

        self.torch = torch
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = _select_transform(transforms, model_type)

    def predict(self, image_path: str) -> np.ndarray:
        """Estimate a relative depth map resized to the original image size."""
        image = _load_rgb_image(image_path)
        image_height, image_width = image.shape[:2]
        input_batch = self.transform(image).to(self.device)

        with self.torch.inference_mode():
            prediction = self.model(input_batch)
            prediction = self.torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(image_height, image_width),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.detach().cpu().numpy().astype(np.float32)


def _select_transform(transforms: Any, model_type: str) -> Any:
    if model_type in {"DPT_Large", "DPT_Hybrid", "DPT_BEiT_L_384"}:
        return transforms.dpt_transform
    return transforms.small_transform


def _load_rgb_image(image_path: str) -> np.ndarray:
    with Image.open(Path(image_path)) as image:
        return np.asarray(image.convert("RGB"))


def estimate_depth(image_path: str) -> np.ndarray:
    """Smoke helper using the default MiDaS relative depth model."""
    return DepthEstimator().predict(image_path)
