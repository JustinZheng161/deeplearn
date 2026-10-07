"""Framework-agnostic regularization helpers for model parameters."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any


def _flatten(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    def visit(value: Any) -> None:
        if isinstance(value, (str, bytes)):
            raise TypeError("parameters must contain only numeric values")
        if isinstance(value, Iterable) and not isinstance(value, (int, float, complex)):
            for child in value:
                visit(child)
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("parameters must contain only numeric values")
        if not math.isfinite(value):
            raise ValueError("parameters must contain finite values")
        result.append(float(value))
    visit(values)
    if not result:
        raise ValueError("parameters must not be empty")
    return result


def l1_penalty(parameters: Iterable[Any], *, coefficient: float = 1.0) -> float:
    """Return coefficient times the sum of absolute parameter values."""
    _validate_coefficient(coefficient)
    return float(coefficient) * sum(abs(value) for value in _flatten(parameters))


def l2_penalty(parameters: Iterable[Any], *, coefficient: float = 1.0) -> float:
    """Return coefficient times the sum of squared parameter values."""
    _validate_coefficient(coefficient)
    return float(coefficient) * sum(value * value for value in _flatten(parameters))


def elastic_net_penalty(
    parameters: Iterable[Any], *, l1_coefficient: float = 0.5, l2_coefficient: float = 0.5
) -> float:
    """Combine L1 sparsity and L2 shrinkage penalties."""
    _validate_coefficient(l1_coefficient, "l1_coefficient")
    _validate_coefficient(l2_coefficient, "l2_coefficient")
    values = _flatten(parameters)
    return float(l1_coefficient) * sum(abs(value) for value in values) + float(l2_coefficient) * sum(value * value for value in values)


def weight_decay_update(
    parameters: Sequence[float], gradients: Sequence[float], *, learning_rate: float, decay: float
) -> list[float]:
    """Apply a decoupled weight-decay update and return new parameters."""
    values = _vector(parameters, "parameters")
    updates = _vector(gradients, "gradients")
    if len(values) != len(updates):
        raise ValueError("parameters and gradients must have the same length")
    _validate_positive(learning_rate, "learning_rate")
    _validate_coefficient(decay, "decay")
    step = float(learning_rate)
    return [(1.0 - step * float(decay)) * value - step * gradient for value, gradient in zip(values, updates)]


def clip_vector_norm(values: Sequence[float], *, max_norm: float) -> list[float]:
    """Scale a vector down to ``max_norm`` while preserving its direction."""
    vector = _vector(values, "values")
    _validate_positive(max_norm, "max_norm")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= max_norm or norm == 0:
        return vector.copy()
    scale = float(max_norm) / norm
    return [value * scale for value in vector]


def _vector(values: Sequence[float], name: str) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence")
    result = list(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    for value in result:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must contain numbers")
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain finite values")
    return [float(value) for value in result]


def _validate_coefficient(value: float, name: str = "coefficient") -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


__all__ = ["clip_vector_norm", "elastic_net_penalty", "l1_penalty", "l2_penalty", "weight_decay_update"]
