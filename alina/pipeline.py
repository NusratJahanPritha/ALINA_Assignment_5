
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from .color import normalize_color_features
from .color_detection import detect_color_masks
from .config import PipelineConfig
from .histogram import calc_histogram
from .roi import roi_points_to_quad
from .traversal import (
    circular_threshold_pixel_discovery_and_traversal,
)


@dataclass
class ImageResult:
    filename: str
    had_lines: bool
    num_pixels: int
    elapsed_seconds: float


def process_image(
    img: np.ndarray,
    roi_points: np.ndarray,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, bool]:

    final_img = img.copy()
    no_lines_img = img.copy()

    # 1. Convert ALINA ROI into four points.
    roi_quad = roi_points_to_quad(
        roi_points, dtype=np.float32
    ).reshape(4, 2)

    # 2. Destination points for bird's-eye transformation.
    dst = np.array(
        [
            config.roi.dst_bottom_left,
            config.roi.dst_top_left,
            config.roi.dst_top_right,
            config.roi.dst_bottom_right,
        ],
        dtype=np.float32,
    ).reshape(4, 2)

    # 3. Perspective transformation.
    M = cv2.getPerspectiveTransform(roi_quad, dst)

    warped = cv2.warpPerspective(
        img,
        M,
        (img.shape[1], img.shape[0]),
    )

    # Save first bird's-eye image for inspection.
    if not process_image._saved_birdseye:
        save_dir = config.output_images_dir.parent / "birdseye"
        save_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(
            str(save_dir / "first_frame_warped.jpg"),
            warped,
        )
        process_image._saved_birdseye = True

    # 4. Select color detection method.
    method = getattr(config, "color_method", "baseline")
    color = getattr(config, "marking_color", "yellow")

    if method == "baseline":
        # Original ALINA manual thresholding.
        color_features = normalize_color_features(warped)

        lower = np.array(
            config.yellow_lower, dtype=np.uint8
        )
        upper = np.array(
            config.yellow_upper, dtype=np.uint8
        )

        if color != "yellow":
            raise ValueError(
                "Original ALINA baseline currently supports "
                "yellow only. Configure a separate white "
                "baseline before using marking_color='white'."
            )

        mask = cv2.inRange(
            color_features, lower, upper
        )

    else:
        if color not in ("yellow", "white"):
            raise ValueError(
                "marking_color must be 'yellow' or 'white'"
            )

        # Restrict detection to the actual warped ROI.
        roi_mask = np.zeros(
            warped.shape[:2], dtype=np.uint8
        )

        cv2.fillPoly(
            roi_mask,
            [np.round(dst).astype(np.int32)],
            255,
        )

        yellow_mask, white_mask = detect_color_masks(
            warped,
            method=method,
            roi_mask=roi_mask,
        )

        # Yellow and white are processed independently.
        if color == "yellow":
            mask = yellow_mask
        else:
            mask = white_mask

    # 5. Preserve ALINA's column exclusion.
    if config.mask_ignore_left_columns > 0:
        mask[
            :, :config.mask_ignore_left_columns
        ] = 0

    # 6. Original ALINA histogram analysis.
    avg_pixel, peak_value = calc_histogram(
        mask,
        min_white_pixels=config.min_white_pixels,
    )

    if (
        avg_pixel is None
        or peak_value <= config.peak_pixel_threshold
    ):
        return (
            no_lines_img,
            np.empty((0, 2), dtype=np.int32),
            False,
        )

    # 7. Original ALINA CIRCLEDAT traversal.
    binary = mask.copy()

    binary[avg_pixel[1], avg_pixel[0]] = 255

    line_marking_pixels = (
        circular_threshold_pixel_discovery_and_traversal(
            binary,
            avg_pixel[0],
            avg_pixel[1],
            config.circular_threshold,
        )
    )

    height, width = binary.shape[:2]
    black_img = np.zeros(
        (height, width), dtype=np.uint8
    )

    for x, y in line_marking_pixels:
        cv2.circle(
            black_img, (x, y), 1, 255, -1
        )

    # 8. Transform detected pixels back to original image.
    Minv = cv2.getPerspectiveTransform(dst, roi_quad)

    unwarped = cv2.warpPerspective(
        black_img,
        Minv,
        (img.shape[1], img.shape[0]),
        flags=cv2.INTER_NEAREST,
    )

    line_pixels_only = np.where(unwarped > 0)

    x_coords = line_pixels_only[1]
    y_coords = line_pixels_only[0]

    coords = np.column_stack(
        (x_coords, y_coords)
    )

    if len(coords) == 0:
        return no_lines_img, coords, False

    # 9. Annotate detected pixels in red.
    final_img[line_pixels_only] = (0, 0, 255)

    return final_img, coords, True


process_image._saved_birdseye = False


def run_batch(
    config: PipelineConfig,
    roi_points: np.ndarray,
) -> list[ImageResult]:

    image_filenames = sorted(
        f
        for f in config.input_dir.iterdir()
        if f.suffix.lower() == ".jpg"
    )

    if not image_filenames:
        raise FileNotFoundError(
            f"No JPG frames in {config.input_dir}"
        )

    total = len(image_filenames)

    process_image._saved_birdseye = False

    # Confirm the ROI on the first frame.
    reference_quad = roi_points_to_quad(
        roi_points, dtype=np.int32
    )

    reference_image = cv2.imread(
        str(image_filenames[0])
    )

    if reference_image is None:
        raise ValueError(
            "Cannot read the reference image"
        )

    from .roi import compute_display_scale

    scale = compute_display_scale(
        reference_image,
        max_display_dim=1080,
    )

    display_image = (
        cv2.resize(
            reference_image,
            None,
            fx=scale,
            fy=scale,
        )
        if scale != 1.0
        else reference_image.copy()
    )

    display_quad = (
        (reference_quad * scale).astype(np.int32)
        if scale != 1.0
        else reference_quad
    )

    cv2.polylines(
        display_image,
        display_quad,
        True,
        (0, 0, 255),
        thickness=2,
    )

    cv2.imshow("Confirm ROI", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Labeling Process Initiated!")

    results: list[ImageResult] = []

    log_file = (
        open(config.log_file, "w")
        if config.log_file
        else None
    )

    try:
        for i, path in enumerate(
            image_filenames, start=1
        ):
            start = time.perf_counter()

            img = cv2.imread(str(path))
            if img is None:
                continue

            annotated, coords, had_lines = process_image(
                img, roi_points, config
            )

            elapsed = time.perf_counter() - start

            np.savetxt(
                config.output_coords_dir
                / f"{path.stem}.txt",
                coords,
                fmt="%6d",
            )

            cv2.imwrite(
                str(config.output_images_dir / path.name),
                annotated,
            )

            status = (
                "Labeled"
                if had_lines
                else "No lines found"
            )

            log_line = (
                f"[{i}/{total}] [{status}] "
                f"{path.name}: {len(coords)} px, "
                f"elapsed {elapsed:.3f}s "
                f"({datetime.now().strftime('%H:%M:%S')})"
            )

            print(log_line)

            if log_file:
                log_file.write(log_line + "\n")
                log_file.flush()

            results.append(
                ImageResult(
                    path.name,
                    had_lines,
                    len(coords),
                    elapsed,
                )
            )

    finally:
        if log_file:
            log_file.close()

    labeled_count = sum(
        r.had_lines for r in results
    )

    total_elapsed = sum(
        r.elapsed_seconds for r in results
    )

    avg_ms = (
        total_elapsed / len(results) * 1000
        if results
        else 0.0
    )

    fps = 1000 / avg_ms if avg_ms else 0.0

    summary = (
        f"Done: {labeled_count}/{len(results)} labeled, "
        f"avg {avg_ms:.1f}ms/frame "
        f"({fps:.2f} fps), "
        f"total {total_elapsed:.1f}s"
    )

    print(summary)

    if config.log_file:
        with open(config.log_file, "a") as f:
            f.write(summary + "\n")

    return results
