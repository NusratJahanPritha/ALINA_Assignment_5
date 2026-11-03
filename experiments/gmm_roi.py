
from pathlib import Path
import json
import time

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_ROOT = Path("data/Raw_Data")
OUTPUT_ROOT = Path("outputs/gmm")

VIDEOS = ["vidd_1", "vidd_2", "vidd_3"]

MAX_DIM = 960

N_COMPONENTS = 5
TRAIN_SAMPLES = 15000
RANDOM_STATE = 42

YELLOW_LOWER = np.array([10, 45, 70], dtype=np.uint8)
YELLOW_UPPER = np.array([45, 255, 255], dtype=np.uint8)

MIN_CLUSTER_YELLOW_SCORE = 0.02


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
# GMM SEGMENTATION
# ==========================================

def create_gmm_mask(image):
    height, width = image.shape[:2]

    # Convert BGR image to LAB color space.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Normalized float64 values improve numerical stability.
    pixels = lab.reshape(-1, 3).astype(np.float64) / 255.0

    rng = np.random.default_rng(RANDOM_STATE)

    sample_count = min(TRAIN_SAMPLES, len(pixels))

    indices = rng.choice(
        len(pixels),
        size=sample_count,
        replace=False
    )

    training_pixels = pixels[indices]

    # Stable GMM configuration.
    gmm = GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type="diag",
        max_iter=100,
        reg_covar=1e-3,
        random_state=RANDOM_STATE,
        n_init=3,
        init_params="kmeans"
    )

    gmm.fit(training_pixels)

    # Predict labels in chunks to reduce peak RAM usage.
    labels = np.empty(len(pixels), dtype=np.uint8)

    chunk_size = 50000

    for start in range(0, len(pixels), chunk_size):
        end = min(start + chunk_size, len(pixels))

        labels[start:end] = gmm.predict(
            pixels[start:end]
        ).astype(np.uint8)

    labels = labels.reshape(height, width)

    # Create HSV-based yellow reference.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(
        hsv,
        YELLOW_LOWER,
        YELLOW_UPPER
    )

    # Exclude the upper 45% of the image.
    yellow_mask[:int(height * 0.45), :] = 0

    # Find the GMM cluster with the highest yellow proportion.
    best_cluster = None
    best_score = -1.0

    for cluster_id in range(N_COMPONENTS):
        cluster_pixels = labels == cluster_id

        total_pixels = int(np.count_nonzero(cluster_pixels))

        if total_pixels == 0:
            continue

        yellow_pixels = int(
            np.count_nonzero(
                cluster_pixels & (yellow_mask > 0)
            )
        )

        score = yellow_pixels / total_pixels

        if score > best_score:
            best_score = score
            best_cluster = cluster_id

    if (
        best_cluster is None
        or best_score < MIN_CLUSTER_YELLOW_SCORE
    ):
        raise RuntimeError(
            "No reliable yellow GMM cluster found. "
            f"Best yellow score: {best_score:.4f}"
        )

    # Retain pixels belonging to the selected cluster
    # that also satisfy the yellow reference threshold.
    final_mask = np.where(
        (labels == best_cluster) & (yellow_mask > 0),
        255,
        0
    ).astype(np.uint8)

    # Remove isolated noise.
    kernel = np.ones((3, 3), dtype=np.uint8)

    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    print(
        f"  Best GMM cluster: {best_cluster}, "
        f"yellow score: {best_score:.4f}, "
        f"mask pixels: {cv2.countNonZero(final_mask)}"
    )

    return final_mask


# ==========================================
# CONVERT MASK TO TRAPEZOIDAL ROI
# ==========================================

def mask_to_roi(mask, original_width, original_height):
    height, width = mask.shape[:2]

    ys, xs = np.where(mask > 0)

    if len(xs) < 30:
        raise RuntimeError(
            "Not enough yellow pixels to estimate ROI."
        )

    # Robust bounds of detected pixels.
    x_min = float(np.percentile(xs, 5))
    x_max = float(np.percentile(xs, 95))

    y_min = float(np.percentile(ys, 5))
    y_max = float(np.percentile(ys, 95))

    # Expand ROI to include surrounding taxiway.
    x_margin = 0.12 * width
    y_margin = 0.20 * height

    left = max(0.0, x_min - x_margin)
    right = min(width - 1.0, x_max + x_margin)

    top = max(height * 0.45, y_min - y_margin)
    bottom = min(height - 1.0, y_max + y_margin)

    # Minimum ROI width and height.
    min_width = 0.08 * width
    min_height = 0.20 * height

    if right - left < min_width:
        center_x = (left + right) / 2.0

        left = max(0.0, center_x - min_width / 2.0)
        right = min(width - 1.0, center_x + min_width / 2.0)

    if bottom - top < min_height:
        center_y = (top + bottom) / 2.0

        top = max(
            height * 0.45,
            center_y - min_height / 2.0
        )

        bottom = min(
            height - 1.0,
            top + min_height
        )

        top = max(
            height * 0.45,
            bottom - min_height
        )

    if right <= left or bottom <= top:
        raise RuntimeError("Invalid ROI geometry.")

    # Make the top of the trapezoid narrower.
    top_inset = 0.15 * (right - left)

    roi = np.array(
        [
            [left, bottom],                # Bottom-left
            [left + top_inset, top],       # Top-left
            [right - top_inset, top],      # Top-right
            [right, bottom]               # Bottom-right
        ],
        dtype=np.float32
    )

    # Convert back to original image coordinates.
    roi[:, 0] *= original_width / width
    roi[:, 1] *= original_height / height

    return roi


# ==========================================
# SAVE ROI PREVIEW
# ==========================================

def save_preview(image, roi, output_path):
    preview = image.copy()

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
            10,
            (0, 255, 0),
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
            f"Could not save preview: {output_path}"
        )


# ==========================================
# PROCESS FIRST FRAME
# ==========================================

def process_video(video):
    video_dir = INPUT_ROOT / video

    frames = sorted(video_dir.glob("*.jpg"))

    if not frames:
        raise FileNotFoundError(
            f"No JPG frames found in {video_dir}"
        )

    first_frame_path = frames[0]

    image = cv2.imread(str(first_frame_path))

    if image is None:
        raise RuntimeError(
            f"Cannot read image: {first_frame_path}"
        )

    original_height, original_width = image.shape[:2]

    print(f"\nProcessing: {video}")
    print(f"First frame: {first_frame_path.name}")
    print(
        f"Resolution: {original_width} x {original_height}"
    )

    start_time = time.perf_counter()

    resized = resize_image(image)

    mask = create_gmm_mask(resized)

    roi = mask_to_roi(
        mask,
        original_width,
        original_height
    )

    inference_seconds = time.perf_counter() - start_time

    roi_path = OUTPUT_ROOT / f"{video}_roi.txt"
    mask_path = OUTPUT_ROOT / f"{video}_mask.jpg"
    preview_path = OUTPUT_ROOT / f"{video}_preview.jpg"

    # Save ROI in the exact format required by
    # your existing run_saved_roi.py:
    # 4 rows, 2 columns: BL, TL, TR, BR.
    np.savetxt(
        roi_path,
        roi,
        fmt="%.2f"
    )

    if not cv2.imwrite(str(mask_path), mask):
        raise RuntimeError(
            f"Could not save mask: {mask_path}"
        )

    save_preview(
        image,
        roi,
        preview_path
    )

    print("ROI points [BL, TL, TR, BR]:")
    print(roi)

    print(
        f"GMM inference time: "
        f"{inference_seconds:.3f} seconds"
    )

    print(f"Saved ROI: {roi_path}")
    print(f"Saved mask: {mask_path}")
    print(f"Saved preview: {preview_path}")

    return {
        "video": video,
        "status": "success",
        "first_frame": first_frame_path.name,
        "image_width": original_width,
        "image_height": original_height,
        "inference_seconds": round(
            inference_seconds, 4
        ),
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

            # Remove stale outputs from earlier runs.
            for suffix in (
                "_roi.txt",
                "_mask.jpg",
                "_preview.jpg"
            ):
                stale_file = OUTPUT_ROOT / f"{video}{suffix}"

                if stale_file.exists():
                    stale_file.unlink()

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

    print("\n================================")
    print("GMM ROI DETECTION COMPLETE")
    print("================================")

    for result in results:
        print(
            f"{result['video']} -> {result['status']}"
        )

    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
