
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alina.color_detection import detect_color_masks
from alina.roi import roi_points_to_quad


METHODS = [
    "kmeans",
    "gmm",
    "dbscan",
    "fuzzy_cmeans",
    "meanshift",
]

VIDEOS = ["vidd_1", "vidd_2", "vidd_3"]


def load_roi_mask(image, roi_path):
    """
    Load the saved Assignment 4 ROI.

    Accepts:
        4 points: BL, TL, TR, BR
        6 points: ALINA's original representation

    Returns:
        Binary ROI mask in original image dimensions.
    """
    h, w = image.shape[:2]

    points = np.loadtxt(
        roi_path, dtype=np.float32
    ).reshape(-1, 2)

    if len(points) == 6:
        quad = roi_points_to_quad(
            points.reshape(1, 6, 2),
            dtype=np.float32,
        ).reshape(4, 2)

    elif len(points) == 4:
        quad = points

    else:
        raise ValueError(
            f"Expected 4 or 6 ROI points: {roi_path}"
        )

    roi_mask = np.zeros(
        (h, w), dtype=np.uint8
    )

    cv2.fillPoly(
        roi_mask,
        [np.round(quad).astype(np.int32)],
        255,
    )

    return roi_mask


def save_preview(image, mask, color, path):
    """
    Highlight detected pixels on the original frame.
    Yellow markings: yellow overlay
    White markings: white overlay
    """
    overlay = image.copy()

    if color == "yellow":
        overlay[mask > 0] = (0, 255, 255)
    else:
        overlay[mask > 0] = (255, 255, 255)

    preview = cv2.addWeighted(
        image, 0.55,
        overlay, 0.45,
        0,
    )

    cv2.imwrite(str(path), preview)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Assignment 5: Detect yellow and white "
            "taxiway markings using one ML method."
        )
    )

    parser.add_argument(
        "--method",
        required=True,
        choices=METHODS,
    )

    parser.add_argument(
        "--video",
        required=True,
        choices=VIDEOS,
    )

    parser.add_argument(
        "--roi-file",
        default=None,
        help=(
            "Optional saved ROI file. "
            "Default: outputs/gmm/<video>_roi.txt"
        ),
    )

    args = parser.parse_args()

    input_dir = (
        ROOT / "data" / "Raw_Data" / args.video
    )

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input folder not found: {input_dir}"
        )

    # Automatically load the same saved ROI
    # for all five methods.
    roi_path = (
        Path(args.roi_file)
        if args.roi_file
        else ROOT / "outputs" / "gmm"
             / f"{args.video}_roi.txt"
    )

    if not roi_path.is_absolute():
        roi_path = ROOT / roi_path

    if not roi_path.is_file():
        raise FileNotFoundError(
            f"ROI file not found: {roi_path}"
        )

    # Assignment 5 outputs are separate
    # from Assignment 4.
    output_dir = (
        ROOT / "outputs" / "assignment_5"
        / args.method / args.video
    )

    yellow_dir = output_dir / "yellow_masks"
    white_dir = output_dir / "white_masks"

    yellow_preview_dir = (
        output_dir / "yellow_previews"
    )

    white_preview_dir = (
        output_dir / "white_previews"
    )

    roi_dir = output_dir / "roi_masks"

    for folder in [
        yellow_dir,
        white_dir,
        yellow_preview_dir,
        white_preview_dir,
        roi_dir,
    ]:
        folder.mkdir(
            parents=True, exist_ok=True
        )

    frames = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() == ".jpg"
    )

    if not frames:
        raise FileNotFoundError(
            f"No JPG frames found: {input_dir}"
        )

    print(f"Method: {args.method}")
    print(f"Video: {args.video}")
    print(f"ROI: {roi_path}")
    print(f"Frames: {len(frames)}")

    results = []

    for i, frame_path in enumerate(
        frames, start=1
    ):
        image = cv2.imread(
            str(frame_path)
        )

        if image is None:
            print(
                f"Skipping unreadable image: "
                f"{frame_path.name}"
            )
            continue

        roi_mask = load_roi_mask(
            image, roi_path
        )

        # Measure color detection only.
        start = time.perf_counter()

        yellow_mask, white_mask = (
            detect_color_masks(
                image,
                method=args.method,
                roi_mask=roi_mask,
            )
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        stem = frame_path.stem

        # Save yellow and white separately.
        cv2.imwrite(
            str(yellow_dir / f"{stem}.png"),
            yellow_mask,
        )

        cv2.imwrite(
            str(white_dir / f"{stem}.png"),
            white_mask,
        )

        cv2.imwrite(
            str(roi_dir / f"{stem}.png"),
            roi_mask,
        )

        # Separate visual previews.
        save_preview(
            image,
            yellow_mask,
            "yellow",
            yellow_preview_dir / f"{stem}.jpg",
        )

        save_preview(
            image,
            white_mask,
            "white",
            white_preview_dir / f"{stem}.jpg",
        )

        yellow_pixels = int(
            np.count_nonzero(yellow_mask)
        )

        white_pixels = int(
            np.count_nonzero(white_mask)
        )

        results.append({
            "frame": frame_path.name,
            "method": args.method,
            "yellow_pixels": yellow_pixels,
            "white_pixels": white_pixels,
            "processing_time_ms": round(
                elapsed_ms, 2
            ),
        })

        print(
            f"[{i}/{len(frames)}] "
            f"{frame_path.name} | "
            f"Yellow: {yellow_pixels} px | "
            f"White: {white_pixels} px | "
            f"Time: {elapsed_ms:.2f} ms"
        )

    if not results:
        raise RuntimeError(
            "No frames were successfully processed."
        )

    # Save per-frame metrics.
    csv_path = output_dir / "results.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "method",
                "yellow_pixels",
                "white_pixels",
                "processing_time_ms",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    average_time = np.mean([
        row["processing_time_ms"]
        for row in results
    ])

    yellow_detected = sum(
        row["yellow_pixels"] > 0
        for row in results
    )

    white_detected = sum(
        row["white_pixels"] > 0
        for row in results
    )

    print("\nExperiment completed.")
    print(f"Method: {args.method}")
    print(f"Processed frames: {len(results)}")
    print(
        f"Frames with yellow pixels: "
        f"{yellow_detected}/{len(results)}"
    )
    print(
        f"Frames with white pixels: "
        f"{white_detected}/{len(results)}"
    )
    print(
        f"Average processing time: "
        f"{average_time:.2f} ms/frame"
    )
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
