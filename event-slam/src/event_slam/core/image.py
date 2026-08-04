from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale or BGR image to unsigned 8-bit grayscale."""
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)

    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Unsupported image shape: {image.shape}")
