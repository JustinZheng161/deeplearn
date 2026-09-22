"""Dependency-free metrics for deep-learning regression tasks."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _values(values: Sequence[float], name: str) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence")
    try:
        result = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a numeric sequence") from exc
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in result):
        raise TypeError(f"{name} must contain only numbers")
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite numbers")
    return [float(value) for value in result]


def _pair(targets: Sequence[float], predictions: Sequence[float]) -> tuple[list[float], list[float]]:
    actual = _values(targets, "targets")
    predicted = _values(predictions, "predictions")
    if len(actual) != len(predicted):
        raise ValueError("targets and predictions must have the same length")
    return actual, predicted


def mean_absolute_error(targets: Sequence[float], predictions: Sequence[float]) -> float:
    """Return the mean absolute difference between targets and predictions."""
    actual, predicted = _pair(targets, predictions)
    return sum(abs(target - prediction) for target, prediction in zip(actual, predicted)) / len(actual)


def mean_squared_error(targets: Sequence[float], predictions: Sequence[float]) -> float:
    """Return the mean squared difference between targets and predictions."""
    actual, predicted = _pair(targets, predictions)
    return sum((target - prediction) ** 2 for target, prediction in zip(actual, predicted)) / len(actual)


def r2_score(targets: Sequence[float], predictions: Sequence[float]) -> float:
    """Return the coefficient of determination.

    For constant targets, a perfect prediction returns ``1.0`` and any error
    returns ``0.0`` rather than producing a division-by-zero result.
    """
    actual, predicted = _pair(targets, predictions)
    mean = sum(actual) / len(actual)
    total = sum((target - mean) ** 2 for target in actual)
    residual = sum((target - prediction) ** 2 for target, prediction in zip(actual, predicted))
    if total == 0:
        return 1.0 if residual == 0 else 0.0
    return 1.0 - residual / total


def huber_loss(actual: float, predicted: float, *, delta: float = 1.0) -> float:
    """Return the Huber loss for one regression prediction.

    Errors within ``delta`` use a quadratic penalty; larger errors use a
    linear penalty, limiting the influence of mislabeled or noisy examples.
    """
    for name, value in (("actual", actual), ("predicted", predicted)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if isinstance(delta, bool) or not isinstance(delta, (int, float)):
        raise TypeError("delta must be a number")
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive and finite")
    error = abs(float(actual) - float(predicted))
    if error <= delta:
        return 0.5 * error * error
    return float(delta) * (error - 0.5 * float(delta))


def huber_loss_batch(
    actual: Sequence[float], predicted: Sequence[float], *, delta: float = 1.0
) -> float:
    """Return the mean Huber loss across a regression batch."""
    if isinstance(actual, (str, bytes)) or isinstance(predicted, (str, bytes)):
        raise TypeError("actual and predicted must be sequences")
    actual_values = list(actual)
    predicted_values = list(predicted)
    if not actual_values or not predicted_values:
        raise ValueError("actual and predicted must not be empty")
    if len(actual_values) != len(predicted_values):
        raise ValueError("actual and predicted must have the same length")
    losses = [huber_loss(left, right, delta=delta)
              for left, right in zip(actual_values, predicted_values)]
    return sum(losses) / len(losses)


__all__ = ["huber_loss", "huber_loss_batch", "mean_absolute_error", "mean_squared_error", "r2_score"]
