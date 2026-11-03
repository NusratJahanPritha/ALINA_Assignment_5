
from pathlib import Path
import json
import time

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans


VIDEOS = ["vidd_1", "vidd_2", "vidd_3"]

INPUT_ROOT = Path("data/Raw_Data")
OUTPUT_DIR = Path("outputs/kmeans")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 16
MAX_DIM = 960
RANDOM_STATE = 42


def detect_roi(image):
    start = time.perf_counter()

    height, width = image.shape[:2]

    # Resize for faster clustering
    scale = min(1.0, MAX_DIM / max(width, height))

    if scale < 1.0:
        small = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )
    else:
        small = image.copy()

    h, w = small.shape[:2]

    # Convert image to LAB color space
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)

    # K-Means clustering
    model = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        batch_size=4096,
        n_init=3
    )

    labels = model.fit_predict(pixels).reshape(h, w)

    # HSV reference for identifying yellow-like clusters
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    yellow_reference = cv2.inRange(
        hsv,
        np.array([10, 45, 70], dtype=np.uint8),
        np.array([45, 255, 255], dtype=np.uint8)
    )

    # Ignore upper part of image (mostly sky)
    yellow_reference[:int(h * 0.45), :] = 0

    if cv2.countNonZero(yellow_reference) < 30:
        raise ValueError("Too few yellow candidate pixels")

    # Score each cluster by its proportion of yellow pixels
    scores = []

    for cluster_id in range(N_CLUSTERS):
        cluster_mask = labels == cluster_id

        cluster_size = np.count_nonzero(cluster_mask)

        yellow_count = np.count_nonzero(
            cluster_mask & (yellow_reference > 0)
        )

        score = yellow_count / max(cluster_size, 1)
        scores.append(score)

    yellow_cluster = int(np.argmax(scores))

    if scores[yellow_cluster] < 0.02:
        raise ValueError(
            "No cluster sufficiently associated with yellow"
        )

    # Combine K-Means prediction with HSV reference
    mask = (labels == yellow_cluster).astype(np.uint8) * 255

    mask = cv2.bitwise_and(mask, yellow_reference)

    print(
        f"Selected cluster: {yellow_cluster}, "
        f"yellow proportion: {scores[yellow_cluster]:.3f}, "
        f"candidate pixels: {cv2.countNonZero(mask)}"
    )

    # Remove isolated noise
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Extract yellow candidate coordinates
    ys, xs = np.where(mask > 0)

    if len(xs) < 100:
        raise ValueError(
            "Not enough yellow pixels after filtering"
        )

    # Robust bounding region
    xmin = float(np.percentile(xs, 5))
    xmax = float(np.percentile(xs, 95))
    ymin = float(np.percentile(ys, 5))
    ymax = float(np.percentile(ys, 95))

    # Expand to include surrounding pavement
    margin_x = 0.12 * (xmax - xmin)
    margin_y = 0.20 * (ymax - ymin)

    xmin = max(0, xmin - margin_x)
    xmax = min(w - 1, xmax + margin_x)
    ymin = max(0, ymin - margin_y)
    ymax = min(h - 1, ymax + margin_y)

    if xmax - xmin < w * 0.08:
        raise ValueError("Detected region too narrow")

    # Expand short regions to a minimum height
    min_height = h * 0.20

    if ymax - ymin < min_height:
        center_y = (ymin + ymax) / 2

        ymin = max(int(h * 0.45), center_y - min_height / 2)
        ymax = min(h - 1, ymin + min_height)

    # Convert bounding box to trapezoid
    inset = 0.15 * (xmax - xmin)

    points_small = np.array([
        [xmin, ymax],           # Bottom-left
        [xmin + inset, ymin],   # Top-left
        [xmax - inset, ymin],   # Top-right
        [xmax, ymax]            # Bottom-right
    ], dtype=np.float32)

    # Convert coordinates back to original resolution
    points = points_small / scale

    elapsed = time.perf_counter() - start

    return points, mask, elapsed


def main():
    results = []

    for video in VIDEOS:
        print(f"\n{'=' * 45}")
        print(f"Processing {video}")
        print("=" * 45)

        # Remove previous outputs to avoid stale results
        for suffix in [
            "_roi.txt",
            "_preview.jpg",
            "_mask.jpg"
        ]:
            (OUTPUT_DIR / f"{video}{suffix}").unlink(
                missing_ok=True
            )

        input_dir = INPUT_ROOT / video

        if not input_dir.is_dir():
            print("Failed: Input folder not found")

            results.append({
                "video": video,
                "status": "failed: input folder not found",
                "inference_seconds": None,
                "roi_points": None
            })
            continue

        frames = sorted(
            p for p in input_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if not frames:
            print("Failed: No frames found")

            results.append({
                "video": video,
                "status": "failed: no frames found",
                "inference_seconds": None,
                "roi_points": None
            })
            continue

        image = cv2.imread(str(frames[0]))

        if image is None:
            print("Failed: Cannot read first frame")

            results.append({
                "video": video,
                "status": "failed: cannot read first frame",
                "inference_seconds": None,
                "roi_points": None
            })
            continue

        try:
            points, mask, elapsed = detect_roi(image)

            # Save ROI coordinates
            np.savetxt(
                OUTPUT_DIR / f"{video}_roi.txt",
                points,
                fmt="%.2f"
            )

            # Create preview
            preview = image.copy()

            cv2.polylines(
                preview,
                [np.round(points).astype(np.int32)],
                True,
                (0, 0, 255),
                5
            )

            for label, (x, y) in zip(
                ["BL", "TL", "TR", "BR"],
                np.round(points).astype(int)
            ):
                cv2.circle(
                    preview,
                    (x, y),
                    8,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    preview,
                    label,
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2
                )

            cv2.imwrite(
                str(OUTPUT_DIR / f"{video}_preview.jpg"),
                preview
            )

            # Save binary mask at clustering resolution
            cv2.imwrite(
                str(OUTPUT_DIR / f"{video}_mask.jpg"),
                mask
            )

            print("ROI coordinates:")
            print(points)

            print(f"Inference time: {elapsed:.3f} seconds")
            print("Status: SUCCESS")

            results.append({
                "video": video,
                "status": "valid",
                "inference_seconds": elapsed,
                "roi_points": points.tolist()
            })

        except Exception as e:
            print(f"Status: FAILED - {e}")

            results.append({
                "video": video,
                "status": f"failed: {e}",
                "inference_seconds": None,
                "roi_points": None
            })

    # Save results
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nAll three videos processed.")
    print("Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
