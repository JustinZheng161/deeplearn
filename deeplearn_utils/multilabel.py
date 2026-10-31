"""Dependency-free metrics for multi-label classification outputs."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _matrix(values: Sequence[Sequence[int | float]], name: str) -> list[list[float]]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of rows")
    rows = [list(row) for row in values]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{name} must be a non-empty rectangular matrix")
    for row in rows:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must contain finite numeric values")
    return [[float(value) for value in row] for row in rows]


def _binary(values: Sequence[Sequence[int | float]], name: str) -> list[list[int]]:
    rows = _matrix(values, name)
    if any(value not in {0.0, 1.0} for row in rows for value in row):
        raise ValueError(f"{name} must contain only zero and one")
    return [[int(value) for value in row] for row in rows]


def _paired(predictions: Sequence[Sequence[int]], targets: Sequence[Sequence[int]]) -> tuple[list[list[int]], list[list[int]]]:
    predicted = _binary(predictions, "predictions")
    expected = _binary(targets, "targets")
    if len(predicted) != len(expected) or len(predicted[0]) != len(expected[0]):
        raise ValueError("predictions and targets must have the same shape")
    return predicted, expected


def threshold_predictions(probabilities: Sequence[Sequence[float]], threshold: float = 0.5) -> list[list[int]]:
    """Convert independent label probabilities to a binary indicator matrix."""
    rows = _matrix(probabilities, "probabilities")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    if any(value < 0 or value > 1 for row in rows for value in row):
        raise ValueError("probabilities must be between zero and one")
    return [[int(value >= threshold) for value in row] for row in rows]


def hamming_loss(predictions: Sequence[Sequence[int]], targets: Sequence[Sequence[int]]) -> float:
    """Return the fraction of incorrectly predicted labels."""
    predicted, expected = _paired(predictions, targets)
    errors = sum(left != right for row_a, row_b in zip(predicted, expected) for left, right in zip(row_a, row_b))
    return errors / (len(predicted) * len(predicted[0]))


def subset_accuracy(predictions: Sequence[Sequence[int]], targets: Sequence[Sequence[int]]) -> float:
    """Return the fraction of rows whose complete label set is correct."""
    predicted, expected = _paired(predictions, targets)
    return sum(left == right for left, right in zip(predicted, expected)) / len(predicted)


def jaccard_score(predictions: Sequence[Sequence[int]], targets: Sequence[Sequence[int]], *, average: str = "samples") -> float:
    """Return intersection-over-union with sample or label averaging."""
    predicted, expected = _paired(predictions, targets)
    if average not in {"samples", "labels"}:
        raise ValueError("average must be 'samples' or 'labels'")
    if average == "samples":
        scores = []
        for left, right in zip(predicted, expected):
            intersection = sum(a == b == 1 for a, b in zip(left, right))
            union = sum(a == 1 or b == 1 for a, b in zip(left, right))
            scores.append(intersection / union if union else 1.0)
        return sum(scores) / len(scores)
    scores = []
    for column in range(len(predicted[0])):
        intersection = sum(row[column] == target[column] == 1 for row, target in zip(predicted, expected))
        union = sum(row[column] == 1 or target[column] == 1 for row, target in zip(predicted, expected))
        scores.append(intersection / union if union else 1.0)
    return sum(scores) / len(scores)


def multilabel_macro_f1(
    predictions: Sequence[Sequence[int]], targets: Sequence[Sequence[int]]
) -> float:
    """Return the unweighted mean F1 across independent labels.

    Each label receives equal weight, which prevents common labels from
    dominating evaluation when a multi-label dataset is imbalanced.
    """
    predicted, expected = _paired(predictions, targets)
    scores = []
    for column in range(len(predicted[0])):
        true_positive = sum(row[column] == target[column] == 1 for row, target in zip(predicted, expected))
        false_positive = sum(row[column] == 1 and target[column] == 0 for row, target in zip(predicted, expected))
        false_negative = sum(row[column] == 0 and target[column] == 1 for row, target in zip(predicted, expected))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def micro_f1(predictions: Sequence[Sequence[int]], targets: Sequence[Sequence[int]]) -> float:
    """Return F1 after pooling decisions over all samples and labels."""
    predicted, expected = _paired(predictions, targets)
    true_positive = sum(a == b == 1 for row, target in zip(predicted, expected) for a, b in zip(row, target))
    false_positive = sum(a == 1 and b == 0 for row, target in zip(predicted, expected) for a, b in zip(row, target))
    false_negative = sum(a == 0 and b == 1 for row, target in zip(predicted, expected) for a, b in zip(row, target))
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 1.0


__all__ = ["hamming_loss", "jaccard_score", "micro_f1", "multilabel_macro_f1", "subset_accuracy", "threshold_predictions"]
