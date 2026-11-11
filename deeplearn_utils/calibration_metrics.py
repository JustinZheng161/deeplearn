"""Calibration metrics for probabilistic classification outputs."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _validate_binary(correct: Sequence[bool], confidences: Sequence[float]) -> tuple[list[bool], list[float]]:
    if isinstance(correct, (str, bytes)) or isinstance(confidences, (str, bytes)):
        raise TypeError("correct and confidences must be sequences")
    outcomes, values = list(correct), list(confidences)
    if not outcomes or len(outcomes) != len(values):
        raise ValueError("correct and confidences must be non-empty and have the same length")
    if any(not isinstance(value, bool) for value in outcomes):
        raise TypeError("correct values must be booleans")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("confidences must contain finite numbers")
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidences must be between zero and one")
    return outcomes, [float(value) for value in values]


def _validate_bins(n_bins: int) -> None:
    if isinstance(n_bins, bool) or not isinstance(n_bins, int):
        raise TypeError("n_bins must be an integer")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")


def adaptive_calibration_error(
    correct: Sequence[bool], confidences: Sequence[float], *, n_bins: int = 10
) -> float:
    """Return ECE using equal-count, confidence-sorted bins.

    Adaptive bins are useful when confidence values are concentrated in a
    narrow range: every non-empty bin has roughly the same sample support,
    unlike fixed-width bins that can be mostly empty.
    """
    outcomes, values = _validate_binary(correct, confidences)
    _validate_bins(n_bins)
    order = sorted(range(len(values)), key=values.__getitem__)
    bin_count = min(n_bins, len(order))
    total_error = 0.0
    for bin_index in range(bin_count):
        start = bin_index * len(order) // bin_count
        end = (bin_index + 1) * len(order) // bin_count
        members = order[start:end]
        accuracy = sum(outcomes[index] for index in members) / len(members)
        confidence = sum(values[index] for index in members) / len(members)
        total_error += len(members) / len(order) * abs(accuracy - confidence)
    return total_error


def classwise_calibration_error(
    labels: Sequence[int], probabilities: Sequence[Sequence[float]], *, n_bins: int = 10
) -> list[float]:
    """Return one-vs-rest adaptive ECE for every class column.

    ``labels`` contains integer class indices and each probability row must
    have the same number of classes. The result preserves column order, so it
    can be averaged or inspected to identify classes needing recalibration.
    """
    _validate_bins(n_bins)
    if isinstance(labels, (str, bytes)) or isinstance(probabilities, (str, bytes)):
        raise TypeError("labels and probabilities must be sequences")
    targets, rows = list(labels), list(probabilities)
    if not targets or len(targets) != len(rows):
        raise ValueError("labels and probabilities must be non-empty and aligned")
    class_count = len(rows[0])
    if class_count < 2:
        raise ValueError("probability rows must contain at least two classes")
    for label in targets:
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < class_count:
            raise ValueError("labels must be valid integer class indices")
    for row in rows:
        if isinstance(row, (str, bytes)) or len(row) != class_count:
            raise ValueError("all probability rows must have the same class count")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in row):
            raise ValueError("probabilities must contain finite numbers")
        if any(value < 0 or value > 1 for value in row) or not math.isclose(sum(row), 1.0, abs_tol=1e-8):
            raise ValueError("each probability row must sum to one")
    return [
        adaptive_calibration_error(
            [label == class_index for label in targets],
            [row[class_index] for row in rows],
            n_bins=n_bins,
        )
        for class_index in range(class_count)
    ]


__all__ = ["adaptive_calibration_error", "classwise_calibration_error"]
