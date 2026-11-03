from __future__ import annotations

from itertools import product

import numpy as np


def circular_threshold_pixel_discovery_and_traversal(img: np.ndarray, x: int, y: int, threshold: int) -> list[tuple[int, int]]:
    visited = {(x, y)}
    line_marking_pixels = []
    stack = [(x, y)]

    circular_range = list(product(range(-threshold, threshold + 1), repeat=2))
    circular_range.remove((0, 0))

    height, width = img.shape[:2]

    while stack:
        cx, cy = stack.pop()

        if img[cy][cx] != 255:
            continue

        line_marking_pixels.append((cx, cy))

        for dx, dy in circular_range:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited:
                continue
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            stack.append((nx, ny))
            visited.add((nx, ny))

    return line_marking_pixels