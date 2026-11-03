
from pathlib import Path
import json
import time

import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_ROOT = Path("data/Raw_Data")
OUTPUT_ROOT = Path("outputs/hough_ransac")

VIDEOS = ["vidd_1", "vidd_2", "vidd_3"]

MAX_DIM = 960
RANDOM_STATE = 42

# HSV yellow reference (OpenCV HSV ranges)
YELLOW_LOWER = np.array([10, 45, 70], dtype=np.uint8)
YELLOW_UPPER = np.array([45, 255, 255], dtype=np.uint8)

GROUND_START = 0.45

# Hough transform settings
HOUGH_THRESHOLD = 15
MIN_LINE_LENGTH = 15
MAX_LINE_GAP = 30

# ROI expansion
X_MARGIN_RATIO = 0.12
Y_MARGIN_RATIO = 0.20
MIN_ROI_WIDTH_RATIO = 0.12
MIN_ROI_HEIGHT_RATIO = 0.20
TOP_INSET_RATIO = 0.15


# ==========================================
# IMAGE RESIZING
# ==========================================

def resize_image(image):
    height, width = image.shape[:2]

    scale = min(1.0, MAX_DIM / max(width, height))

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


# ==========================================
# HSV + CANNY + HOUGH
# ==========================================

def detect_hough_lines(image):
    height, width = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(
        hsv,
        YELLOW_LOWER,
        YELLOW_UPPER
    )

    # Ignore the sky and upper image region.
    yellow_mask[:int(height * GROUND_START), :] = 0

    # Reduce small isolated noise.
    kernel = np.ones((3, 3), dtype=np.uint8)

    yellow_mask = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Detect edges of yellow markings.
    edges = cv2.Canny(
        yellow_mask,
        threshold1=50,
        threshold2=150
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP
    )

    if lines is None:
        raise RuntimeError(
            "HoughLinesP detected no yellow line segments."
        )

    segments = lines.reshape(-1, 4)

    # Reject segments outside the yellow marking region.
    valid_segments = []

    for x1, y1, x2, y2 in segments:
        length = np.hypot(x2 - x1, y2 - y1)

        if length < MIN_LINE_LENGTH:
            continue

        valid_segments.append([x1, y1, x2, y2])

    if not valid_segments:
        raise RuntimeError(
            "No valid Hough line segments found."
        )

    return (
        yellow_mask,
        edges,
        np.asarray(valid_segments, dtype=np.int32)
    )


# ==========================================
# RANSAC LINE FITTING
# ==========================================

def fit_ransac_line(segments):
    """
    Fit x as a function of y.

    This is suitable for approximately vertical
    taxiway centerline markings.
    """

    points = []

    for x1, y1, x2, y2 in segments:
        # Sample points along each Hough segment.
        length = np.hypot(x2 - x1, y2 - y1)

        n = max(2, int(length / 4))

        xs = np.linspace(x1, x2, n)
        ys = np.linspace(y1, y2, n)

        points.extend(zip(xs, ys))

    points = np.asarray(points, dtype=np.float64)

    if len(points) < 10:
        raise RuntimeError(
            "Not enough points for RANSAC."
        )

    # Input: y coordinate
    # Target: x coordinate
    X = points[:, 1].reshape(-1, 1)
    y = points[:, 0]

    model = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=2,
        residual_threshold=20.0,
        max_trials=100,
        random_state=RANDOM_STATE
    )

    model.fit(X, y)

    inliers = model.inlier_mask_

    if inliers is None or np.count_nonzero(inliers) < 10:
        raise RuntimeError(
            "RANSAC could not identify enough inliers."
        )

    return model, points[inliers]


# ==========================================
# BUILD TRAPEZOIDAL ROI
# ==========================================

def build_roi(
    yellow_mask,
    inlier_points,
    original_width,
    original_height
):
    height, width = yellow_mask.shape[:2]

    # Prefer the RANSAC-supported line geometry.
    xs = inlier_points[:, 0]
    ys = inlier_points[:, 1]

    if len(xs) < 10:
        raise RuntimeError(
            "Insufficient RANSAC inlier points."
        )

    x_min = float(np.percentile(xs, 5))
    x_max = float(np.percentile(xs, 95))

    y_min = float(np.percentile(ys, 5))
    y_max = float(np.percentile(ys, 95))

    # Include yellow pixels close to the fitted line.
    # This helps cover the full width of the marking.
    mask_ys, mask_xs = np.where(yellow_mask > 0)

    if len(mask_xs) >= 30:
        x_min = min(
            x_min,
            float(np.percentile(mask_xs, 5))
        )
        x_max = max(
            x_max,
            float(np.percentile(mask_xs, 95))
        )

        y_min = min(
            y_min,
            float(np.percentile(mask_ys, 5))
        )
        y_max = max(
            y_max,
            float(np.percentile(mask_ys, 95))
        )

    left = max(
        0.0,
        x_min - X_MARGIN_RATIO * width
    )

    right = min(
        width - 1.0,
        x_max + X_MARGIN_RATIO * width
    )

    top = max(
        GROUND_START * height,
        y_min - Y_MARGIN_RATIO * height
    )

    bottom = min(
        height - 1.0,
        y_max + Y_MARGIN_RATIO * height
    )

    # Ensure minimum width.
    min_width = MIN_ROI_WIDTH_RATIO * width

    if right - left < min_width:
        center_x = (left + right) / 2.0

        left = max(
            0.0,
            center_x - min_width / 2.0
        )

        right = min(
            width - 1.0,
            center_x + min_width / 2.0
        )

    # Ensure minimum height.
    min_height = MIN_ROI_HEIGHT_RATIO * height

    if bottom - top < min_height:
        center_y = (top + bottom) / 2.0

        top = max(
            GROUND_START * height,
            center_y - min_height / 2.0
        )

        bottom = min(
            height - 1.0,
            top + min_height
        )

        top = max(
            GROUND_START * height,
            bottom - min_height
        )

    if right <= left or bottom <= top:
        raise RuntimeError(
            "Invalid trapezoid dimensions."
        )

    top_inset = TOP_INSET_RATIO * (right - left)

    roi = np.array(
        [
            [left, bottom],              # Bottom-left
            [left + top_inset, top],     # Top-left
            [right - top_inset, top],    # Top-right
            [right, bottom]             # Bottom-right
        ],
        dtype=np.float32
    )

    # Convert to original resolution.
    roi[:, 0] *= original_width / width
    roi[:, 1] *= original_height / height

    return roi


# ==========================================
# SAVE PREVIEW
# ==========================================

def save_preview(
    image,
    roi,
    segments,
    scale_x,
    scale_y,
    output_path
):
    preview = image.copy()

    # Hough segments in green.
    for x1, y1, x2, y2 in segments:
        cv2.line(
            preview,
            (
                int(round(x1 * scale_x)),
                int(round(y1 * scale_y))
            ),
            (
                int(round(x2 * scale_x)),
                int(round(y2 * scale_y))
            ),
            (0, 255, 0),
            3
        )

    # ROI in red.
    polygon = np.round(roi).astype(np.int32)

    cv2.polylines(
        preview,
        [polygon],
        isClosed=True,
        color=(0, 0, 255),
        thickness=5
    )

    for index, point in enumerate(polygon):
        x, y = map(int, point)

        cv2.circle(
            preview,
            (x, y),
            9,
            (255, 0, 0),
            -1
        )

        cv2.putText(
            preview,
            str(index),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(
            f"Failed to save preview: {output_path}"
        )


# ==========================================
# PROCESS EACH VIDEO
# ==========================================

def process_video(video):
    video_dir = INPUT_ROOT / video

    frames = sorted(video_dir.glob("*.jpg"))

    if not frames:
        raise FileNotFoundError(
            f"No JPG frames in {video_dir}"
        )

    first_frame_path = frames[0]

    image = cv2.imread(str(first_frame_path))

    if image is None:
        raise RuntimeError(
            f"Cannot read {first_frame_path}"
        )

    original_height, original_width = image.shape[:2]

    print(f"\nProcessing {video}")
    print(f"First frame: {first_frame_path.name}")
    print(
        f"Resolution: {original_width} x {original_height}"
    )

    start_time = time.perf_counter()

    resized = resize_image(image)

    yellow_mask, edges, segments = detect_hough_lines(
        resized
    )

    model, inlier_points = fit_ransac_line(segments)

    roi = build_roi(
        yellow_mask,
        inlier_points,
        original_width,
        original_height
    )

    inference_seconds = time.perf_counter() - start_time

    roi_path = OUTPUT_ROOT / f"{video}_roi.txt"
    mask_path = OUTPUT_ROOT / f"{video}_mask.jpg"
    edges_path = OUTPUT_ROOT / f"{video}_edges.jpg"
    preview_path = OUTPUT_ROOT / f"{video}_preview.jpg"

    np.savetxt(
        roi_path,
        roi,
        fmt="%.2f"
    )

    cv2.imwrite(str(mask_path), yellow_mask)
    cv2.imwrite(str(edges_path), edges)

    scale_x = original_width / resized.shape[1]
    scale_y = original_height / resized.shape[0]

    save_preview(
        image,
        roi,
        segments,
        scale_x,
        scale_y,
        preview_path
    )

    print(f"Hough segments: {len(segments)}")
    print(f"RANSAC inlier points: {len(inlier_points)}")
    print("ROI points [BL, TL, TR, BR]:")
    print(roi)

    print(
        f"Inference time: "
        f"{inference_seconds:.3f} seconds"
    )

    return {
        "video": video,
        "status": "success",
        "first_frame": first_frame_path.name,
        "image_width": original_width,
        "image_height": original_height,
        "inference_seconds": round(
            inference_seconds, 4
        ),
        "hough_segments": len(segments),
        "ransac_inlier_points": len(inlier_points),
        "roi_points": roi.tolist()
    }


# ==========================================
# MAIN
# ==========================================

def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    for video in VIDEOS:
        try:
            result = process_video(video)

        except Exception as error:
            print(
                f"\nERROR processing {video}: {error}"
            )

            # Remove stale files from previous runs.
            for suffix in (
                "_roi.txt",
                "_mask.jpg",
                "_edges.jpg",
                "_preview.jpg"
            ):
                path = OUTPUT_ROOT / f"{video}{suffix}"

                if path.exists():
                    path.unlink()

            result = {
                "video": video,
                "status": "failed",
                "error": str(error)
            }

        results.append(result)

    results_path = OUTPUT_ROOT / "results.json"

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=2
        )

    print("\n====================================")
    print("HOUGH + RANSAC ROI DETECTION COMPLETE")
    print("====================================")

    for result in results:
        print(
            f"{result['video']} -> {result['status']}"
        )

    print(f"\nResults: {results_path}")


if __name__ == "__main__":
    main()
