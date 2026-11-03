# from __future__ import annotations

# from pathlib import Path

# import cv2
# import numpy as np


# def calc_histogram(
#     img: np.ndarray,
#     min_white_pixels: int = 200,
#     save_path: str | Path | None = None
# ) -> tuple[list[int], int]:

#     # Count white pixels in each column without
#     # allocating a full-size int64 array.
#     projection = cv2.reduce(
#         img, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S
#     ).ravel() // 255

#     if save_path is not None:
#         from matplotlib import pyplot as plt

#         plt.figure()
#         plt.plot(projection)
#         plt.xlabel("X-coordinate")
#         plt.ylabel("Number of white pixels")
#         plt.savefig(str(save_path))
#         plt.close()

#     peak_value = int(np.max(projection))
#     peak_index = int(np.argmax(projection))

#     white_pixels = np.flatnonzero(
#         img[:, peak_index] == 255
#     )

#     if len(white_pixels) > min_white_pixels:
#         avg_pixel = [
#             peak_index,
#             int(np.mean(white_pixels))
#         ]
#         return avg_pixel, peak_value

#     return [], peak_value











from __future__ import annotations

from pathlib import Path
from statistics import mean

import numpy as np


def calc_histogram(img: np.ndarray, min_white_pixels: int = 200, save_path: str | Path | None = None) -> tuple[list[int], int]:
    binary_img = np.where(img == 255, 1, 0)
    projection = np.sum(binary_img, axis=0)

    if save_path is not None:
        from matplotlib import pyplot as plt

        plt.figure()
        plt.plot(projection)
        plt.xlabel("X-coordinate")
        plt.ylabel("Number of white pixels")
        plt.savefig(str(save_path))
        plt.close()

    peak_value = int(np.max(projection))
    peak_index = int(np.argmax(projection))

    peak_area = binary_img[:, peak_index]
    white_pixels = [i for i, row in enumerate(peak_area) if row == 1]

    if len(white_pixels) > min_white_pixels:
        avg_pixel = [peak_index, int(mean(white_pixels))]
        return avg_pixel, peak_value

    return [], peak_value