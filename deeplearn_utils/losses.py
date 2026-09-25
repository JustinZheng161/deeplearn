"""Dependency-light losses for classification and segmentation experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _probability(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return float(value)


def binary_focal_loss(
    targets: Sequence[int], probabilities: Sequence[float], *, gamma: float = 2.0, alpha: float = 0.25, epsilon: float = 1e-15
) -> float:
    """Return mean binary focal loss from labels and positive probabilities."""
    if isinstance(targets, (str, bytes)) or isinstance(probabilities, (str, bytes)):
        raise TypeError("targets and probabilities must be sequences")
    labels = list(targets)
    scores = list(probabilities)
    if not labels or len(labels) != len(scores):
        raise ValueError("targets and probabilities must have the same non-zero length")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be a finite non-negative number")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be between zero and one half")
    losses = []
    for target, probability in zip(labels, scores):
        if isinstance(target, bool) or target not in {0, 1}:
            raise ValueError("targets must contain only zero or one")
        probability = _probability(probability, "probabilities")
        clipped = min(max(probability, epsilon), 1.0 - epsilon)
        true_probability = clipped if target else 1.0 - clipped
        weight = alpha if target else 1.0 - alpha
        losses.append(-weight * (1.0 - true_probability) ** float(gamma) * math.log(true_probability))
    return sum(losses) / len(losses)


def multiclass_focal_loss(
    targets: Sequence[int], probabilities: Sequence[Sequence[float]], *, gamma: float = 2.0, alpha: float = 1.0, epsilon: float = 1e-15
) -> float:
    """Return mean focal loss for normalized multiclass probabilities."""
    labels = list(targets)
    rows = [list(row) for row in probabilities]
    if not labels or len(labels) != len(rows):
        raise ValueError("targets and probabilities must have the same non-zero length")
    if gamma < 0 or not math.isfinite(gamma) or alpha <= 0 or not math.isfinite(alpha):
        raise ValueError("gamma must be non-negative and alpha must be positive")
    losses = []
    class_count = None
    for target, row in zip(labels, rows):
        if isinstance(target, bool) or not isinstance(target, int):
            raise TypeError("targets must contain integer class indices")
        if not row:
            raise ValueError("probability rows must not be empty")
        class_count = class_count or len(row)
        if len(row) != class_count or target < 0 or target >= len(row):
            raise ValueError("invalid class index or inconsistent class count")
        values = [_probability(value, "probabilities") for value in row]
        if not math.isclose(sum(values), 1.0, abs_tol=1e-8):
            raise ValueError("each probability row must sum to one")
        true_probability = min(max(values[target], epsilon), 1.0 - epsilon)
        losses.append(-float(alpha) * (1.0 - true_probability) ** float(gamma) * math.log(true_probability))
    return sum(losses) / len(losses)


def dice_coefficient(
    targets: Sequence[float], probabilities: Sequence[float], *, smooth: float = 1.0
) -> float:
    """Return soft Dice overlap for binary masks or flattened probabilities."""
    labels = list(targets)
    scores = list(probabilities)
    if not labels or len(labels) != len(scores):
        raise ValueError("targets and probabilities must have the same non-zero length")
    if not math.isfinite(smooth) or smooth < 0:
        raise ValueError("smooth must be finite and non-negative")
    target_values = [_probability(value, "targets") for value in labels]
    probability_values = [_probability(value, "probabilities") for value in scores]
    intersection = sum(target * probability for target, probability in zip(target_values, probability_values))
    denominator = sum(target * target for target in target_values) + sum(value * value for value in probability_values)
    return (2.0 * intersection + smooth) / (denominator + smooth)


def dice_loss(
    targets: Sequence[float], probabilities: Sequence[float], *, smooth: float = 1.0
) -> float:
    """Return one minus :func:`dice_coefficient`."""
    return 1.0 - dice_coefficient(targets, probabilities, smooth=smooth)


__all__ = ["binary_focal_loss", "dice_coefficient", "dice_loss", "multiclass_focal_loss"]
