"""Numerically stable probability transforms for model outputs."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _values(values: Sequence[float], name: str = "values") -> list[float]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence")
    try:
        result = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a numeric sequence") from exc
    if not result:
        raise ValueError(f"{name} must not be empty")
    for value in result:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must contain numbers")
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain finite values")
    return [float(value) for value in result]


def sigmoid(value: float) -> float:
    """Return a numerically stable logistic sigmoid."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("value must be a number")
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def softmax(logits: Sequence[float]) -> list[float]:
    """Convert logits to a probability vector using max-shift stabilization."""
    values = _values(logits, "logits")
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def log_softmax(logits: Sequence[float]) -> list[float]:
    """Return log probabilities using the log-sum-exp trick."""
    values = _values(logits, "logits")
    maximum = max(values)
    log_total = maximum + math.log(sum(math.exp(value - maximum) for value in values))
    return [value - log_total for value in values]


def softmax_batch(logits: Sequence[Sequence[float]]) -> list[list[float]]:
    """Apply :func:`softmax` independently to each batch row."""
    if isinstance(logits, (str, bytes)):
        raise TypeError("logits must be a sequence of rows")
    try:
        rows = list(logits)
    except TypeError as exc:
        raise TypeError("logits must be a sequence of rows") from exc
    if not rows:
        raise ValueError("logits must not be empty")
    return [softmax(row) for row in rows]


def categorical_entropy(probabilities: Sequence[float], *, base: float = math.e) -> float:
    """Return Shannon entropy, optionally measured in bits with ``base=2``."""
    values = _values(probabilities, "probabilities")
    if isinstance(base, bool) or not isinstance(base, (int, float)):
        raise TypeError("base must be a number")
    if not math.isfinite(base) or base <= 0 or base == 1:
        raise ValueError("base must be positive and different from one")
    if any(value < 0 for value in values):
        raise ValueError("probabilities must be non-negative")
    total = sum(values)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("probabilities must sum to one")
    return -sum(value * math.log(value, base) for value in values if value > 0)


def binary_cross_entropy(target: float, probability: float, *, epsilon: float = 1e-15) -> float:
    """Return clipped binary cross-entropy for one target/probability pair."""
    for name, value in (("target", target), ("probability", probability)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if target < 0 or target > 1:
        raise ValueError("target must be between zero and one")
    if probability < 0 or probability > 1:
        raise ValueError("probability must be between zero and one")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a number")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be between zero and one half")
    clipped = min(max(float(probability), epsilon), 1.0 - epsilon)
    return -(float(target) * math.log(clipped) + (1.0 - float(target)) * math.log(1.0 - clipped))


def brier_score(
    target: Sequence[float], probabilities: Sequence[float]
) -> float:
    """Return the multiclass Brier score for one probability distribution.

    The score is the sum of squared differences between target and predicted
    probabilities; lower values indicate better calibrated predictions.
    Soft targets are accepted for label smoothing and mixup workflows.
    """
    targets = _values(target, "target")
    predicted = _values(probabilities, "probabilities")
    if len(targets) != len(predicted):
        raise ValueError("target and probabilities must have the same length")
    if any(value < 0 for value in targets):
        raise ValueError("target values must be non-negative")
    if any(value < 0 or value > 1 for value in predicted):
        raise ValueError("probabilities must be between zero and one")
    if not math.isclose(sum(targets), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("target values must sum to one")
    if not math.isclose(sum(predicted), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("probabilities must sum to one")
    return sum((target_value - probability) ** 2
               for target_value, probability in zip(targets, predicted))


def brier_score_batch(
    targets: Sequence[Sequence[float]],
    probabilities: Sequence[Sequence[float]],
) -> float:
    """Return the mean multiclass Brier score across a batch."""
    if isinstance(targets, (str, bytes)) or isinstance(probabilities, (str, bytes)):
        raise TypeError("targets and probabilities must be sequences of rows")
    target_rows = list(targets)
    probability_rows = list(probabilities)
    if not target_rows or not probability_rows:
        raise ValueError("targets and probabilities must not be empty")
    if len(target_rows) != len(probability_rows):
        raise ValueError("targets and probabilities must have the same batch size")
    scores = [brier_score(target, prediction)
              for target, prediction in zip(target_rows, probability_rows)]
    return sum(scores) / len(scores)


def categorical_cross_entropy(
    target: Sequence[float],
    probabilities: Sequence[float],
    *,
    epsilon: float = 1e-15,
) -> float:
    """Return cross-entropy for one normalized target distribution.

    Soft targets are supported, which makes the function compatible with
    label smoothing and mixup. Probabilities are clipped only for the
    logarithm, so small numerical underflow cannot produce infinite loss.
    """
    targets = _values(target, "target")
    predicted = _values(probabilities, "probabilities")
    if len(targets) != len(predicted):
        raise ValueError("target and probabilities must have the same length")
    if any(value < 0 for value in targets):
        raise ValueError("target values must be non-negative")
    if any(value < 0 or value > 1 for value in predicted):
        raise ValueError("probabilities must be between zero and one")
    if not math.isclose(sum(targets), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("target values must sum to one")
    if not math.isclose(sum(predicted), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("probabilities must sum to one")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a number")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be between zero and one half")
    return -sum(target_value * math.log(max(probability, epsilon))
                for target_value, probability in zip(targets, predicted)
                if target_value > 0)


def categorical_cross_entropy_batch(
    targets: Sequence[Sequence[float]],
    probabilities: Sequence[Sequence[float]],
    *,
    epsilon: float = 1e-15,
) -> float:
    """Return the mean categorical cross-entropy across a batch."""
    if isinstance(targets, (str, bytes)) or isinstance(probabilities, (str, bytes)):
        raise TypeError("targets and probabilities must be sequences of rows")
    target_rows = list(targets)
    probability_rows = list(probabilities)
    if not target_rows or not probability_rows:
        raise ValueError("targets and probabilities must not be empty")
    if len(target_rows) != len(probability_rows):
        raise ValueError("targets and probabilities must have the same batch size")
    losses = [categorical_cross_entropy(target, prediction, epsilon=epsilon)
              for target, prediction in zip(target_rows, probability_rows)]
    return sum(losses) / len(losses)


__all__ = [
    "binary_cross_entropy",
    "brier_score",
    "brier_score_batch",
    "categorical_cross_entropy",
    "categorical_cross_entropy_batch",
    "categorical_entropy",
    "log_softmax",
    "sigmoid",
    "softmax",
    "softmax_batch",
]
