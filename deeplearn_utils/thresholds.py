"""Threshold selection helpers for binary model probabilities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ThresholdResult:
    """Best threshold and the score achieved on the supplied validation set."""

    threshold: float
    score: float
    objective: str
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


def _validate(probabilities: Sequence[float], labels: Sequence[int]) -> tuple[list[float], list[int]]:
    if isinstance(probabilities, (str, bytes)) or isinstance(labels, (str, bytes)):
        raise TypeError("probabilities and labels must be sequences")
    try:
        scores = list(probabilities)
        targets = list(labels)
    except TypeError as exc:
        raise TypeError("probabilities and labels must be sequences") from exc
    if not scores:
        raise ValueError("probabilities and labels must not be empty")
    if len(scores) != len(targets):
        raise ValueError("probabilities and labels must have the same length")
    for score in scores:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError("probabilities must contain numbers")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("probabilities must be between zero and one")
    for label in targets:
        if isinstance(label, bool) or not isinstance(label, int) or label not in {0, 1}:
            raise ValueError("labels must contain only zero or one")
    return [float(score) for score in scores], targets


def confusion_at_threshold(
    probabilities: Sequence[float], labels: Sequence[int], threshold: float = 0.5
) -> tuple[int, int, int, int]:
    """Return ``(TP, FP, TN, FN)`` for a probability threshold."""
    scores, targets = _validate(probabilities, labels)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a number")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    predicted = [score >= threshold for score in scores]
    tp = sum(prediction and target == 1 for prediction, target in zip(predicted, targets))
    fp = sum(prediction and target == 0 for prediction, target in zip(predicted, targets))
    tn = sum(not prediction and target == 0 for prediction, target in zip(predicted, targets))
    fn = sum(not prediction and target == 1 for prediction, target in zip(predicted, targets))
    return tp, fp, tn, fn


def _score(tp: int, fp: int, tn: int, fn: int, objective: str) -> float:
    if objective == "f1":
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if objective == "balanced_accuracy":
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        return (sensitivity + specificity) / 2
    if objective == "youden":
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        return sensitivity + specificity - 1.0
    raise ValueError("objective must be 'f1', 'balanced_accuracy', or 'youden'")


def best_threshold(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    objective: str = "f1",
) -> ThresholdResult:
    """Select the validation threshold maximizing a classification objective.

    Only thresholds at observed probability values plus the two boundary
    thresholds are evaluated. Ties prefer the conventional threshold ``0.5``
    and then the lower threshold, making results deterministic.
    """
    scores, targets = _validate(probabilities, labels)
    if objective not in {"f1", "balanced_accuracy", "youden"}:
        raise ValueError("objective must be 'f1', 'balanced_accuracy', or 'youden'")
    candidates = sorted({0.0, 0.5, 1.0, *scores})
    best: ThresholdResult | None = None
    for threshold in candidates:
        tp, fp, tn, fn = confusion_at_threshold(scores, targets, threshold)
        score = _score(tp, fp, tn, fn, objective)
        result = ThresholdResult(threshold, score, objective, tp, fp, tn, fn)
        if best is None or score > best.score + 1e-15:
            best = result
        elif math.isclose(score, best.score) and abs(threshold - 0.5) < abs(best.threshold - 0.5):
            best = result
    assert best is not None
    return best


def binary_predictions(probabilities: Sequence[float], threshold: float = 0.5) -> list[int]:
    """Convert probabilities to zero/one predictions at ``threshold``."""
    scores, _ = _validate(probabilities, [0] * len(probabilities))
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a number")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    return [int(score >= threshold) for score in scores]


__all__ = ["ThresholdResult", "best_threshold", "binary_predictions", "confusion_at_threshold"]
