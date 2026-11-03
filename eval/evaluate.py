from __future__ import annotations

import numpy as np

from alina.config import EvaluationConfig

from .metrics import calculate_f1, calculate_precision, calculate_recall


def _load_coords(path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    x = np.zeros(len(lines), dtype=np.int32)
    y = np.zeros(len(lines), dtype=np.int32)
    for i, line in enumerate(lines):
        xi, yi = line.strip().split()
        x[i], y[i] = int(xi), int(yi)
    return x, y


def run_evaluation(config: EvaluationConfig) -> dict[str, float]:
    if len(config.canny_dirs) != len(config.alina_dirs):
        raise ValueError("canny_dirs and alina_dirs must have the same length (paired by position)")

    recall_values: list[float] = []
    precision_values: list[float] = []
    f1_values: list[float] = []

    for canny_dir, alina_dir in zip(config.canny_dirs, config.alina_dirs):
        for txt_file in sorted(canny_dir.glob("*.txt")):
            matching = alina_dir / txt_file.name
            if not matching.exists():
                print(f"Skipping {txt_file.name}: no matching file in {alina_dir}")
                continue

            x1, y1 = _load_coords(txt_file)
            x2, y2 = _load_coords(matching)

            recall = (calculate_recall(x1, x2) + calculate_recall(y1, y2)) / 2
            precision = (calculate_precision(x1, x2) + calculate_precision(y1, y2)) / 2
            f1 = calculate_f1(precision, recall)

            recall_values.append(recall)
            precision_values.append(precision)
            f1_values.append(f1)

    avg_recall = float(np.mean(recall_values)) if recall_values else 0.0
    avg_precision = float(np.mean(precision_values)) if precision_values else 0.0
    avg_f1 = float(np.mean(f1_values)) if f1_values else 0.0

    print(f"Average Recall: {avg_recall}%")
    print(f"Average Precision: {avg_precision}%")
    print(f"Average F1: {avg_f1}%")

    return {"avg_recall": avg_recall, "avg_precision": avg_precision, "avg_f1": avg_f1}