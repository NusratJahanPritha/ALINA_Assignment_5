from __future__ import annotations

from collections.abc import Iterable


def calculate_recall(ground_truth: Iterable[int], predicted: Iterable[int]) -> float:
    gt, pred = set(ground_truth), set(predicted)
    true_positive = gt & pred
    false_negative = gt - pred
    denom = len(true_positive) + len(false_negative)
    return (len(true_positive) / denom) * 100 if denom else 0.0


def calculate_precision(ground_truth: Iterable[int], predicted: Iterable[int]) -> float:
    gt, pred = set(ground_truth), set(predicted)
    true_positive = gt & pred
    false_positive = pred - gt
    denom = len(true_positive) + len(false_positive)
    return (len(true_positive) / denom) * 100 if denom else 0.0


def calculate_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)