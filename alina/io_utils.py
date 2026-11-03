from __future__ import annotations

import time
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def video_to_frames(input_path: str | Path, output_dir: str | Path, max_consecutive_failures: int = 5) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    cap = cv2.VideoCapture(str(input_path))
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Number of frames: {video_length}")

    count = 0
    malformed = 0
    consecutive_failures = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            malformed += 1
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                break
            continue

        consecutive_failures = 0
        count += 1
        cv2.imwrite(str(output_dir / f"{count:05d}.jpg"), frame)
        if count >= video_length:
            break

    cap.release()
    elapsed = time.time() - start
    print(f"Done: {count} frames extracted, {malformed} malformed, {elapsed:.1f}s")
    return count


def rotate_frames(input_dir: str | Path, output_dir: str | Path, angle: int = 180) -> int:
    rotate_code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    if angle not in rotate_code:
        raise ValueError(f"Unsupported angle {angle}; choose from {sorted(rotate_code)}")

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    start = time.time()
    for image_path in sorted(input_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        rotated = cv2.rotate(img, rotate_code[angle])
        cv2.imwrite(str(output_dir / image_path.name), rotated)
        count += 1

    elapsed = time.time() - start
    print(f"Rotated {count} frames in {elapsed:.1f}s")
    return count


def resize_images(input_dir: str | Path, output_dir: str | Path, width: int = 1920, height: int = 1080) -> int:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for image_path in sorted(input_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        resized = cv2.resize(img, (width, height))
        cv2.imwrite(str(output_dir / image_path.name), resized)
        count += 1

    print(f"Resized {count} images to {width}x{height}")
    return count