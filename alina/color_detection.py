
from __future__ import annotations

import cv2
import numpy as np

from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

SEED = 42
MAX_DIM = 480
SAMPLE_SIZE = 6000

METHODS = (
    "kmeans",
    "gmm",
    "dbscan",
    "fuzzy_cmeans",
    "meanshift",
)


def _resize(image, max_dim=MAX_DIM):
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / max(h, w))

    if scale == 1.0:
        return image.copy()

    return cv2.resize(
        image,
        (max(1, round(w * scale)),
         max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _lab_features(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).astype(np.float32) / 255.0


def _nearest_labels(features, centers, chunk=30000):
    labels = np.empty(len(features), dtype=np.int32)

    for start in range(0, len(features), chunk):
        end = min(start + chunk, len(features))

        distances = np.sum(
            (features[start:end, None, :]
             - centers[None, :, :]) ** 2,
            axis=2,
        )

        labels[start:end] = np.argmin(distances, axis=1)

    return labels


def _fuzzy_centers(sample, k=12, iterations=40):
    rng = np.random.default_rng(SEED)

    indices = rng.choice(
        len(sample), size=min(k, len(sample)), replace=False
    )
    centers = sample[indices].copy()

    for _ in range(iterations):
        dist = np.linalg.norm(
            sample[:, None, :] - centers[None, :, :],
            axis=2,
        )
        dist = np.maximum(dist, 1e-6)

        membership = 1.0 / (dist ** 2)
        membership /= membership.sum(axis=1, keepdims=True)

        weights = membership ** 2
        updated = (
            weights.T @ sample
        ) / np.maximum(weights.sum(axis=0)[:, None], 1e-9)

        if np.allclose(centers, updated, atol=1e-4):
            centers = updated
            break

        centers = updated

    return centers


def _cluster(features, method):
    rng = np.random.default_rng(SEED)

    count = min(SAMPLE_SIZE, len(features))
    sample_ids = rng.choice(
        len(features), count, replace=False
    )
    sample = features[sample_ids]

    if method == "kmeans":
        model = KMeans(
            n_clusters=12,
            random_state=SEED,
            n_init=10,
        )
        model.fit(sample)
        return model.predict(features)

    if method == "gmm":
        model = GaussianMixture(
            n_components=12,
            covariance_type="diag",
            reg_covar=1e-4,
            random_state=SEED,
            max_iter=100,
        )
        model.fit(sample)
        return model.predict(features)

    if method == "fuzzy_cmeans":
        centers = _fuzzy_centers(sample)
        return _nearest_labels(features, centers)

    if method == "meanshift":
        model = MeanShift(
            bandwidth=0.12,
            bin_seeding=True,
            cluster_all=False,
        )
        model.fit(sample)
        return _nearest_labels(
            features, model.cluster_centers_
        )

    if method == "dbscan":
        # DBSCAN cannot predict new points directly.
        model = DBSCAN(
            eps=0.055,
            min_samples=12,
            n_jobs=2,
        )
        sample_labels = model.fit_predict(sample)

        neighbors = NearestNeighbors(
            n_neighbors=1,
            n_jobs=2,
        )
        neighbors.fit(sample)

        labels = np.empty(len(features), dtype=np.int32)

        for start in range(0, len(features), 30000):
            end = min(start + 30000, len(features))
            dist, ids = neighbors.kneighbors(
                features[start:end]
            )

            predicted = sample_labels[ids[:, 0]].copy()
            predicted[dist[:, 0] > 0.055] = -1
            labels[start:end] = predicted

        return labels

    raise ValueError(f"Unknown method: {method}")


def _select_color_clusters(image, labels, valid):
    """
    Classify clusters using their mean BGR color.

    Yellow and white definitions here are broad,
    fixed semantic rules, not learned thresholds.
    """
    h, w = image.shape[:2]
    flat_image = image.reshape(-1, 3)
    flat_valid = valid.reshape(-1) > 0

    yellow = np.zeros(h * w, dtype=np.uint8)
    white = np.zeros(h * w, dtype=np.uint8)

    for cluster_id in np.unique(labels):
        if cluster_id < 0:
            continue

        selected = (labels == cluster_id) & flat_valid

        if np.count_nonzero(selected) < 5:
            continue

        mean_bgr = flat_image[selected].mean(axis=0)

        pixel = np.clip(
            np.round(mean_bgr), 0, 255
        ).astype(np.uint8).reshape(1, 1, 3)

        hsv = cv2.cvtColor(
            pixel, cv2.COLOR_BGR2HSV
        )[0, 0]

        hue, saturation, value = map(int, hsv)

        # Broad yellow color family
        if 15 <= hue <= 40 and saturation >= 55 and value >= 65:
            yellow[selected] = 255

        # Broad white color family
        elif saturation <= 55 and value >= 175:
            white[selected] = 255

    return yellow.reshape(h, w), white.reshape(h, w)


def _filter_components(mask, valid):
    """
    Remove small noise and very large regions.

    Geometric filtering is a heuristic and may
    remove valid wide markings.
    """
    mask = cv2.bitwise_and(mask, valid)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    result = np.zeros_like(mask)
    valid_area = max(1, np.count_nonzero(valid))

    for i in range(1, n):
        x, y, w, h, area = stats[i]

        if area < 5:
            continue

        # Reject very large color regions.
        if area > 0.10 * valid_area:
            continue

        result[labels == i] = 255

    return result


def detect_color_masks(image, method="kmeans", roi_mask=None):
    """
    Input:
        image: BGR image
        method: one of the five clustering algorithms
        roi_mask: optional uint8 mask; nonzero = pavement ROI

    Returns:
        yellow_mask, white_mask
        Both uint8, same dimensions as input, values 0/255.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty input image")

    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")

    original_h, original_w = image.shape[:2]

    small = _resize(image)
    h, w = small.shape[:2]

    if roi_mask is None:
        valid = np.full((h, w), 255, dtype=np.uint8)
    else:
        if roi_mask.shape[:2] != image.shape[:2]:
            raise ValueError("ROI mask dimensions must match image")

        valid = cv2.resize(
            roi_mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        valid = np.where(valid > 0, 255, 0).astype(
            np.uint8
        )

    features = _lab_features(small)
    labels = _cluster(features, method)

    yellow, white = _select_color_clusters(
        small, labels, valid
    )

    yellow = _filter_components(yellow, valid)
    white = _filter_components(white, valid)

    yellow = cv2.resize(
        yellow,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )
    white = cv2.resize(
        white,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )

    # Enforce original-resolution ROI.
    if roi_mask is not None:
        yellow[roi_mask == 0] = 0
        white[roi_mask == 0] = 0

    return yellow, white


def detect_color_mask(image, method="kmeans", roi_mask=None):
    yellow, white = detect_color_masks(
        image, method=method, roi_mask=roi_mask
    )
    return cv2.bitwise_or(yellow, white)
