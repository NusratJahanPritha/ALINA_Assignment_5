from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from alina.roi import compute_display_scale


def _draw_contour_and_canny(img: np.ndarray, init_contour: np.ndarray, filename: str, output_dir: Path) -> None:
    # 4.3 step 1: outline the contour region around the taxiway line marking
    mask = np.zeros_like(img[:, :, 0])
    cv2.fillPoly(mask, np.int32([init_contour]), 255)

    # 4.3 step 2: preprocessing -> grayscale + Gaussian blur to reduce noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    gray_masked = cv2.GaussianBlur(gray_masked, (5, 5), 0)

    # 4.3 step 3: Canny edge detection
    canny = cv2.Canny(gray_masked, 30, 150)
    cv2.polylines(canny, np.int32([init_contour]), True, (0, 0, 0), thickness=2)

    scale = compute_display_scale(canny, max_display_dim=1080)
    display_canny = cv2.resize(canny, None, fx=scale, fy=scale) if scale != 1.0 else canny
    cv2.imshow("CBEM", display_canny)
    cv2.waitKey()

    line_pixels_only = np.where(canny > 0)
    coords = np.column_stack((line_pixels_only[1], line_pixels_only[0]))

    image_basename = Path(filename).stem
    cbem_image_dir = output_dir / "cbem_images"
    cbem_coords_dir = output_dir / "cbem_textfiles"
    cbem_image_dir.mkdir(parents=True, exist_ok=True)
    cbem_coords_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(cbem_image_dir / f"{image_basename}_cbem.jpg"), canny)
    np.savetxt(cbem_coords_dir / f"{image_basename}.txt", coords, fmt="%6d")
    print(f"Saved CBEM for {filename}")


def create_cbem(image_path: str | Path, output_dir: str | Path) -> None:
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    scale = compute_display_scale(img, max_display_dim=1080)
    display_img = cv2.resize(img, None, fx=scale, fy=scale) if scale != 1.0 else img

    drawing = False
    contour_pts: list[tuple[int, int]] = []  # in display-image coordinates
    last_point = None

    def draw_contours(event, x, y, flags, param):
        nonlocal drawing, last_point
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_point = (x, y)
            contour_pts.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.line(display_img, last_point, (x, y), (0, 255, 255), 2)
            last_point = (x, y)
            contour_pts.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.line(display_img, last_point, (x, y), (0, 255, 255), 2)
            contour_pts.append((x, y))

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_contours)

    while True:
        cv2.imshow("image", display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            # rescale the traced contour back to original full-resolution coordinates
            original_scale_contour = [(int(x / scale), int(y / scale)) for x, y in contour_pts]
            init_contour = np.array(original_scale_contour, np.int32)
            _draw_contour_and_canny(cv2.imread(str(image_path)), init_contour, image_path.name, output_dir)
        elif key == 27:  # Esc
            break

    cv2.destroyAllWindows()