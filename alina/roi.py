from __future__ import annotations

import numpy as np
import cv2

POINT_COLOR = (0, 0, 255)
LINE_COLOR = (0, 255, 0)
LINE_THICKNESS = 2


def compute_display_scale(img: np.ndarray, max_display_dim: int) -> float:
    height, width = img.shape[:2]
    largest_dim = max(height, width)
    if largest_dim <= max_display_dim:
        return 1.0
    return max_display_dim / largest_dim


def select_roi(image_path: str, max_display_dim: int = 1080) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    scale = compute_display_scale(img, max_display_dim)
    display_img = cv2.resize(img, None, fx=scale, fy=scale) if scale != 1.0 else img

    roi_img = np.copy(display_img)
    start_point = None
    roi_points: list[tuple[int, int]] = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_points, roi_img, start_point
        if event == cv2.EVENT_LBUTTONDOWN:
            if start_point is None:
                start_point = (x, y)
            else:
                end_point = (x, y)
                roi_points.append(start_point)
                roi_points.append(end_point)
                start_point = end_point

        roi_img = np.copy(display_img)
        if start_point is not None:
            cv2.polylines(roi_img, [np.array(roi_points)], False, POINT_COLOR, LINE_THICKNESS)
            cv2.line(roi_img, start_point, (x, y), LINE_COLOR, LINE_THICKNESS)
        cv2.imshow("Polyline", roi_img)
        
    cv2.namedWindow("Polyline")
    cv2.setMouseCallback("Polyline", mouse_callback)
    cv2.imshow("Polyline", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # rescale clicked points back to the original full-resolution image
    original_scale_points = [(int(x / scale), int(y / scale)) for x, y in roi_points]
    return np.array([original_scale_points], dtype=np.int32)


def roi_points_to_quad(roi_points: np.ndarray, dtype=np.int32) -> np.ndarray:
    pts = roi_points[0]
    return np.array(
        [[
            (pts[0][0], pts[0][1]),
            (pts[1][0], pts[1][1]),
            (pts[3][0], pts[1][1]),
            (pts[5][0], pts[0][1]),
        ]],
        dtype=dtype,
    )