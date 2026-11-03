
from pathlib import Path
import re
import csv

import cv2
import numpy as np

VIDEOS = ["vidd_1", "vidd_2", "vidd_3"]

ALINA_ROOT = Path("outputs/kmeans_alina")
GT_ROOT = Path("data/ground_truth_cbem")
OUTPUT_CSV = Path("outputs/kmeans/evaluation.csv")


def read_alina_coords(path, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)

    text = path.read_text(encoding="utf-8").strip()

    for line in text.splitlines():
        numbers = re.findall(r"-?\d+(?:\.\d+)?", line)

        if len(numbers) < 2:
            continue

        x, y = [int(round(float(v))) for v in numbers[:2]]

        if 0 <= x < width and 0 <= y < height:
            mask[y, x] = 255

    return mask


def evaluate_video(video):
    log_path = ALINA_ROOT / video / "timing.log"
    coords_dir = ALINA_ROOT / video / "coords"
    gt_dir = GT_ROOT / video

    if not log_path.exists():
        print(f"{video}: timing.log missing")
        return None

    text = log_path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    # 1. Average processing time
    records = re.findall(
        r"\[\d+/\d+\]\s*"
        r"\[(Labeled|No lines found)\]"
        r".*?elapsed\s+([\d.]+)s",
        text
    )

    total_frames = len(records)

    labeled_frames = sum(
        status == "Labeled"
        for status, _ in records
    )

    times_ms = [
        float(seconds) * 1000
        for _, seconds in records
    ]

    avg_time = (
        sum(times_ms) / total_frames
        if total_frames else None
    )

    # 2. Frame labeling rate
    labeling_rate = (
        100 * labeled_frames / total_frames
        if total_frames else None
    )

    # 3. Detection rate (Recall), if CBEM exists
    tp = 0
    fn = 0

    if gt_dir.exists():
        for gt_path in sorted(gt_dir.glob("*.png")):
            gt = cv2.imread(
                str(gt_path),
                cv2.IMREAD_GRAYSCALE
            )

            if gt is None:
                continue

            height, width = gt.shape
            gt_mask = gt > 127

            coord_path = coords_dir / f"{gt_path.stem}.txt"

            if not coord_path.exists():
                continue

            pred_mask = read_alina_coords(
                coord_path, width, height
            ) > 0

            tp += int(np.count_nonzero(pred_mask & gt_mask))
            fn += int(np.count_nonzero(~pred_mask & gt_mask))

    recall = (
        100 * tp / (tp + fn)
        if tp + fn > 0 else None
    )

    print(f"\n{video}")
    print(f"Frames labeled: {labeled_frames}/{total_frames}")
    print(f"Frame labeling rate: {labeling_rate}")
    print(f"Average processing time: {avg_time} ms/frame")
    print(f"Detection rate (Recall): {recall}")

    return {
        "video": video,
        "frames_labeled": labeled_frames,
        "total_frames": total_frames,
        "frame_labeling_rate_percent": labeling_rate,
        "detection_rate_recall_percent": recall,
        "average_processing_time_ms": avg_time,
    }


def main():
    results = []

    for video in VIDEOS:
        result = evaluate_video(video)

        if result is not None:
            results.append(result)

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "frames_labeled",
                "total_frames",
                "frame_labeling_rate_percent",
                "detection_rate_recall_percent",
                "average_processing_time_ms",
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
