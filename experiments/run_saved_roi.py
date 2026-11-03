
import argparse
from pathlib import Path

import cv2
import numpy as np

from alina.config import PipelineConfig
from alina.pipeline import run_batch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True,
                        choices=["vidd_1", "vidd_2", "vidd_3"])
    parser.add_argument("--roi-dir",
                        default="outputs/zero_shot_vlm")
    parser.add_argument("--output-root",
                        default="outputs/vlm_alina")

    args = parser.parse_args()

    video = args.video
    input_dir = Path("data/Raw_Data") / video
    roi_file = Path(args.roi_dir) / f"{video}_roi.txt"
    output_dir = Path(args.output_root) / video

    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)

    if not roi_file.is_file():
        raise FileNotFoundError(roi_file)

    frames = sorted(input_dir.glob("*.jpg"))

    if not frames:
        raise FileNotFoundError(
            f"No JPG frames found in {input_dir}"
        )

    image = cv2.imread(str(frames[0]))

    if image is None:
        raise ValueError("Cannot read first frame")

    height, width = image.shape[:2]

    roi_points = np.loadtxt(
        roi_file,
        dtype=np.float32
    ).reshape(-1, 2)

    if roi_points.shape != (4, 2):
        raise ValueError("ROI must contain four points")

    if not np.isfinite(roi_points).all():
        raise ValueError("ROI contains invalid numbers")

    if (
        np.any(roi_points[:, 0] < 0)
        or np.any(roi_points[:, 0] >= width)
        or np.any(roi_points[:, 1] < 0)
        or np.any(roi_points[:, 1] >= height)
    ):
        raise ValueError(
            "ROI coordinates are outside frame bounds. "
            "Check image resolution."
        )

    polygon = roi_points.reshape(-1, 1, 2)

    if (
        not cv2.isContourConvex(polygon)
        or cv2.contourArea(polygon) < width * height * 0.01
    ):
        raise ValueError("Degenerate or very small ROI")

    print(f"Video: {video}")
    print(f"First frame: {frames[0].name}")
    print(f"Image dimensions: {width} x {height}")
    print("Loaded VLM ROI:")
    print(roi_points)

    config = PipelineConfig(
        input_dir=input_dir,
        output_images_dir=output_dir / "annotated",
        output_coords_dir=output_dir / "coords",
        log_file=output_dir / "timing.log",
        peak_pixel_threshold=20,
        min_white_pixels=50,
        mask_ignore_left_columns=0,
    )

    # Uses ALINA's original labeling pipeline
    # Convert 4 VLM points to ALINA's 6-point format
    bl, tl, tr, br = roi_points

    alina_roi_points = np.array([[
        bl,
        tl,
        tl,
        tr,
        tr,
        br
    ]], dtype=np.float32)

    run_batch(config, alina_roi_points)



if __name__ == "__main__":
    main()
