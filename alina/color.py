from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def normalize_color_features(image: np.ndarray, save_path: str | Path | None = None) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX)
    s = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX)
    v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    color_features = np.stack((h, s, v), axis=-1)

    if save_path is not None:
        cv2.imwrite(str(save_path), color_features)

    return color_features