
from pathlib import Path
import json
import time

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# =====================================================
# CONFIGURATION
# =====================================================

INPUT_ROOT = Path("data/Raw_Data")
MASK_ROOT = Path("data/rf_training_masks")
OUTPUT_ROOT = Path("outputs/random_forest")

VIDEOS = ["vidd_1", "vidd_2", "vidd_3"]

MAX_DIM = 960
RANDOM_STATE = 42

# Training samples per video
POSITIVE_SAMPLES = 3000
NEGATIVE_SAMPLES = 6000

# Random Forest configuration
N_ESTIMATORS = 100
MAX_DEPTH = 15
MIN_SAMPLES_LEAF = 3

# Pixel probability threshold
PROBABILITY_THRESHOLD = 0.50

# Restrict predictions to likely ground area
GROUND_START = 0.45

# ROI construction
X_MARGIN_RATIO = 0.12
Y_MARGIN_RATIO = 0.20
MIN_ROI_WIDTH_RATIO = 0.08
MIN_ROI_HEIGHT_RATIO = 0.20
TOP_INSET_RATIO = 0.15


# =====================================================
# IMAGE HELPERS
# =====================================================

def first_frame_path(video):
    folder = INPUT_ROOT / video
    frames = sorted(folder.glob("*.jpg"))

    if not frames:
        raise FileNotFoundError(
            f"No JPG frames found in {folder}"
        )

    return frames[0]


def resize_image(image):
    height, width = image.shape[:2]

    scale = min(
        1.0,
        MAX_DIM / max(width, height)
    )

    new_width = max(
        1,
        int(round(width * scale))
    )

    new_height = max(
        1,
        int(round(height * scale))
    )

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(image):
    """
    Extract color and position features.

    Features:
        B, G, R
        H, S, V
        L, A, B (LAB)
        normalized x coordinate
        normalized y coordinate
    """

    height, width = image.shape[:2]

    bgr = image.astype(np.float32) / 255.0

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    ).astype(np.float32)

    hsv[:, :, 0] /= 179.0
    hsv[:, :, 1:] /= 255.0

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    ).astype(np.float32) / 255.0

    yy, xx = np.indices(
        (height, width),
        dtype=np.float32
    )

    xx /= max(1, width - 1)
    yy /= max(1, height - 1)

    features = np.concatenate(
        [
            bgr,
            hsv,
            lab,
            xx[:, :, None],
            yy[:, :, None]
        ],
        axis=2
    )

    return features.reshape(-1, 11)


# =====================================================
# LOAD TRAINING DATA
# =====================================================

def load_training_samples(video, rng):
    frame_path = first_frame_path(video)

    image = cv2.imread(str(frame_path))

    if image is None:
        raise RuntimeError(
            f"Cannot read {frame_path}"
        )

    mask_path = MASK_ROOT / f"{video}.png"

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Training mask missing: {mask_path}"
        )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise RuntimeError(
            f"Cannot read training mask: {mask_path}"
        )

    # Resize mask to match original frame if needed.
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    image = resize_image(image)

    mask = cv2.resize(
        mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    features = extract_features(image)

    labels = (mask.reshape(-1) > 127).astype(np.uint8)

    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)

    if len(positive_indices) == 0:
        raise RuntimeError(
            f"{video}: training mask has no white pixels."
        )

    if len(negative_indices) == 0:
        raise RuntimeError(
            f"{video}: training mask has no background pixels."
        )

    selected_positive = rng.choice(
        positive_indices,
        size=min(
            POSITIVE_SAMPLES,
            len(positive_indices)
        ),
        replace=False
    )

    selected_negative = rng.choice(
        negative_indices,
        size=min(
            NEGATIVE_SAMPLES,
            len(negative_indices)
        ),
        replace=False
    )

    selected = np.concatenate(
        [selected_positive, selected_negative]
    )

    print(
        f"{video}: "
        f"{len(selected_positive)} positive, "
        f"{len(selected_negative)} negative samples"
    )

    return features[selected], labels[selected]


# =====================================================
# TRAIN RANDOM FOREST
# =====================================================

def train_random_forest():
    rng = np.random.default_rng(RANDOM_STATE)

    feature_batches = []
    label_batches = []

    for video in VIDEOS:
        features, labels = load_training_samples(
            video,
            rng
        )

        feature_batches.append(features)
        label_batches.append(labels)

    X_train = np.vstack(feature_batches)
    y_train = np.concatenate(label_batches)

    print(
        f"\nTraining Random Forest on "
        f"{len(y_train)} labeled pixels..."
    )

    start_time = time.perf_counter()

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=2
    )

    model.fit(X_train, y_train)

    training_seconds = (
        time.perf_counter() - start_time
    )

    print(
        f"Training completed in "
        f"{training_seconds:.3f} seconds"
    )

    return model, training_seconds


# =====================================================
# PREDICT YELLOW-LINE MASK
# =====================================================

def predict_mask(model, image):
    height, width = image.shape[:2]

    features = extract_features(image)

    predictions = np.zeros(
        len(features),
        dtype=np.uint8
    )

    chunk_size = 50000

    positive_class_index = int(
        np.where(model.classes_ == 1)[0][0]
    )

    for start in range(0, len(features), chunk_size):
        end = min(
            start + chunk_size,
            len(features)
        )

        probabilities = model.predict_proba(
            features[start:end]
        )[:, positive_class_index]

        predictions[start:end] = (
            probabilities >= PROBABILITY_THRESHOLD
        ).astype(np.uint8) * 255

    mask = predictions.reshape(height, width)

    # Ignore the upper 45% of the image.
    mask[:int(height * GROUND_START), :] = 0

    # Remove small isolated regions.
    kernel = np.ones((3, 3), dtype=np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return mask


# =====================================================
# MASK TO TRAPEZOIDAL ROI
# =====================================================

def mask_to_roi(
    mask,
    original_width,
    original_height
):
    height, width = mask.shape[:2]

    ys, xs = np.where(mask > 0)

    if len(xs) < 30:
        raise RuntimeError(
            "Not enough predicted yellow pixels."
        )

    x_min = float(np.percentile(xs, 5))
    x_max = float(np.percentile(xs, 95))

    y_min = float(np.percentile(ys, 5))
    y_max = float(np.percentile(ys, 95))

    left = max(
        0.0,
        x_min - X_MARGIN_RATIO * width
    )

    right = min(
        width - 1.0,
        x_max + X_MARGIN_RATIO * width
    )

    top = max(
        GROUND_START * height,
        y_min - Y_MARGIN_RATIO * height
    )

    bottom = min(
        height - 1.0,
        y_max + Y_MARGIN_RATIO * height
    )

    min_width = MIN_ROI_WIDTH_RATIO * width

    if right - left < min_width:
        center_x = (left + right) / 2.0

        left = max(
            0.0,
            center_x - min_width / 2.0
        )

        right = min(
            width - 1.0,
            center_x + min_width / 2.0
        )

    min_height = MIN_ROI_HEIGHT_RATIO * height

    if bottom - top < min_height:
        center_y = (top + bottom) / 2.0

        top = max(
            GROUND_START * height,
            center_y - min_height / 2.0
        )

        bottom = min(
            height - 1.0,
            top + min_height
        )

        top = max(
            GROUND_START * height,
            bottom - min_height
        )

    if right <= left or bottom <= top:
        raise RuntimeError(
            "Invalid ROI geometry."
        )

    top_inset = TOP_INSET_RATIO * (right - left)

    roi = np.array(
        [
            [left, bottom],
            [left + top_inset, top],
            [right - top_inset, top],
            [right, bottom]
        ],
        dtype=np.float32
    )

    roi[:, 0] *= original_width / width
    roi[:, 1] *= original_height / height

    return roi


# =====================================================
# SAVE PREVIEW
# =====================================================

def save_preview(image, roi, output_path):
    preview = image.copy()

    polygon = np.round(roi).astype(np.int32)

    cv2.polylines(
        preview,
        [polygon],
        isClosed=True,
        color=(0, 0, 255),
        thickness=5
    )

    for index, point in enumerate(polygon):
        x, y = map(int, point)

        cv2.circle(
            preview,
            (x, y),
            10,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            preview,
            str(index),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(
            f"Failed to save preview: {output_path}"
        )


# =====================================================
# PROCESS VIDEO
# =====================================================

def process_video(video, model):
    frame_path = first_frame_path(video)

    image = cv2.imread(str(frame_path))

    if image is None:
        raise RuntimeError(
            f"Cannot read {frame_path}"
        )

    original_height, original_width = image.shape[:2]

    print(f"\nProcessing {video}")
    print(f"First frame: {frame_path.name}")

    start_time = time.perf_counter()

    resized = resize_image(image)

    mask = predict_mask(model, resized)

    roi = mask_to_roi(
        mask,
        original_width,
        original_height
    )

    inference_seconds = (
        time.perf_counter() - start_time
    )

    roi_path = OUTPUT_ROOT / f"{video}_roi.txt"
    mask_path = OUTPUT_ROOT / f"{video}_mask.jpg"
    preview_path = OUTPUT_ROOT / f"{video}_preview.jpg"

    np.savetxt(
        roi_path,
        roi,
        fmt="%.2f"
    )

    if not cv2.imwrite(str(mask_path), mask):
        raise RuntimeError(
            f"Failed to save mask: {mask_path}"
        )

    save_preview(
        image,
        roi,
        preview_path
    )

    print("ROI points [BL, TL, TR, BR]:")
    print(roi)

    print(
        f"ROI inference time: "
        f"{inference_seconds:.3f} seconds"
    )

    return {
        "video": video,
        "status": "success",
        "first_frame": frame_path.name,
        "image_width": original_width,
        "image_height": original_height,
        "inference_seconds": round(
            inference_seconds, 4
        ),
        "roi_points": roi.tolist()
    }


# =====================================================
# MAIN
# =====================================================

def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    model, training_seconds = train_random_forest()

    results = []

    for video in VIDEOS:
        try:
            result = process_video(
                video,
                model
            )

        except Exception as error:
            print(
                f"\nERROR processing {video}: {error}"
            )

            for suffix in (
                "_roi.txt",
                "_mask.jpg",
                "_preview.jpg"
            ):
                path = OUTPUT_ROOT / f"{video}{suffix}"

                if path.exists():
                    path.unlink()

            result = {
                "video": video,
                "status": "failed",
                "error": str(error)
            }

        results.append(result)

    with open(
        OUTPUT_ROOT / "results.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "training_seconds": round(
                    training_seconds, 4
                ),
                "results": results
            },
            f,
            indent=2
        )

    print("\n====================================")
    print("RANDOM FOREST ROI DETECTION COMPLETE")
    print("====================================")

    for result in results:
        print(
            f"{result['video']} -> {result['status']}"
        )

    print(
        f"\nResults saved to: "
        f"{OUTPUT_ROOT / 'results.json'}"
    )


if __name__ == "__main__":
    main()
