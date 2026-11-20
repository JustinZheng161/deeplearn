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


def log_cosh_loss(actual: float, predicted: float) -> float:
    """Return the stable log-cosh loss for one regression prediction."""
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise TypeError("actual must be a number")
    if isinstance(predicted, bool) or not isinstance(predicted, (int, float)):
        raise TypeError("predicted must be a number")
    if not math.isfinite(actual) or not math.isfinite(predicted):
        raise ValueError("actual and predicted must be finite")
    error = abs(float(predicted) - float(actual))
    if error > 20:
        return error - math.log(2.0)
    return math.log(math.cosh(error))


def log_cosh_loss_batch(targets: Sequence[float], predictions: Sequence[float]) -> float:
    """Return the mean log-cosh loss across a regression batch."""
    actual, predicted = _pair(targets, predictions)
    return sum(log_cosh_loss(left, right) for left, right in zip(actual, predicted)) / len(actual)


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


def quantile_loss(actual: float, predicted: float, *, quantile: float = 0.5) -> float:
    """Return pinball loss for one target and predicted quantile.

    Under-predictions receive weight ``quantile`` while over-predictions
    receive weight ``1 - quantile``. At ``quantile=0.5`` this is half the
    absolute error, which is useful for median regression.
    """
    for name, value in (("actual", actual), ("predicted", predicted)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
        raise TypeError("quantile must be a number")
    if not math.isfinite(quantile) or not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between zero and one")
    error = float(actual) - float(predicted)
    return float(quantile) * error if error >= 0 else (float(quantile) - 1.0) * error


def quantile_loss_batch(
    actual: Sequence[float],
    predicted: Sequence[float] | Sequence[Sequence[float]],
    *,
    quantile: float = 0.5,
) -> float:
    """Return mean pinball loss for scalar or multi-quantile predictions.

    A one-dimensional prediction sequence uses one shared ``quantile``. For
    multiple quantiles, each prediction row must align with ``quantiles`` and
    the mean is taken over all target-quantile pairs.
    """
    targets = _values(actual, "actual")
    if isinstance(predicted, (str, bytes)):
        raise TypeError("predicted must be a numeric sequence")
    predictions = list(predicted)
    if not predictions:
        raise ValueError("predicted must not be empty")
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in predictions):
        if len(targets) != len(predictions):
            raise ValueError("actual and predicted must have the same length")
        return sum(quantile_loss(left, right, quantile=quantile)
                   for left, right in zip(targets, predictions)) / len(targets)
    rows = [list(row) for row in predictions]
    if len(rows) != len(targets) or not rows or any(not row for row in rows):
        raise ValueError("actual and predicted rows must be non-empty and aligned")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all predicted rows must have the same quantile count")
    if isinstance(quantile, (int, float)) and not isinstance(quantile, bool):
        quantiles = [float(quantile)] * width
    else:
        quantiles = list(quantile)
    if len(quantiles) != width:
        raise ValueError("quantile count must match predicted row width")
    losses = [quantile_loss(target, prediction, quantile=level)
              for target, row in zip(targets, rows) for prediction, level in zip(row, quantiles)]
    return sum(losses) / len(losses)


__all__ = [
    "huber_loss",
    "huber_loss_batch",
    "log_cosh_loss",
    "log_cosh_loss_batch",
    "mean_absolute_error",
    "mean_squared_error",
    "quantile_loss",
    "quantile_loss_batch",
    "r2_score",
]
