from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from alina.roi import compute_display_scale


def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    # Eq. 11/12: auto-threshold Canny based on median pixel intensity
    intensity = np.median(image)
    lower = int(max(0, (1.0 - sigma) * intensity))
    upper = int(min(255, (1.0 + sigma) * intensity))
    return cv2.Canny(image, lower, upper)


def overlay_coords(
    image_path: str | Path,
    coords_path: str | Path,
    save_to_disk: bool = False,
    output_dir: str | Path | None = None,
) -> None:
    image_path, coords_path = Path(image_path), Path(coords_path)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # preprocessing -> grayscale + blur, same as CBEM step 2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = auto_canny(blurred)

    lines = coords_path.read_text().splitlines()
    points = np.zeros((len(lines), 2), dtype=np.int32)
    for i, line in enumerate(lines):
        x, y = line.strip().split()
        points[i] = [int(x), int(y)]

    # draw the given coordinates on both the edge map and the original frame
    overlay_on_original = img.copy()
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    for x, y in points:
        cv2.circle(overlay_on_original, (int(x), int(y)), 2, (0, 0, 255), -1)
        cv2.circle(edges_color, (int(x), int(y)), 2, (0, 0, 255), -1)

    if save_to_disk:
        if output_dir is None:
            raise ValueError("output_dir is required when save_to_disk=True")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "canny_edges.jpg"), edges)
        cv2.imwrite(str(output_dir / "superimposed_on_canny.jpg"), edges_color)
        cv2.imwrite(str(output_dir / "superimposed_on_original.jpg"), overlay_on_original)

    scale = compute_display_scale(edges, max_display_dim=1080)
    display_edges = cv2.resize(edges, None, fx=scale, fy=scale) if scale != 1.0 else edges
    display_edges_color = cv2.resize(edges_color, None, fx=scale, fy=scale) if scale != 1.0 else edges_color

    cv2.imshow("Canny Edges", display_edges)
    cv2.imshow("Superimposed on canny image", display_edges_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()