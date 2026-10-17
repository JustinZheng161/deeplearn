"""Dependency-free activation functions for neural-network experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _values(values: Sequence[float]) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise TypeError("values must be a numeric sequence")
    result = list(values)
    if not result:
        raise ValueError("values must not be empty")
    for value in result:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("values must contain numbers")
        if not math.isfinite(value):
            raise ValueError("values must contain finite values")
    return [float(value) for value in result]


def relu(values: Sequence[float]) -> list[float]:
    """Return max(0, x) elementwise."""
    return [max(0.0, value) for value in _values(values)]


def relu_derivative(values: Sequence[float]) -> list[float]:
    """Return a subgradient of ReLU, choosing zero at the kink."""
    return [1.0 if value > 0 else 0.0 for value in _values(values)]


def leaky_relu(values: Sequence[float], *, negative_slope: float = 0.01) -> list[float]:
    """Return leaky ReLU with a configurable negative slope."""
    _validate_slope(negative_slope)
    return [value if value >= 0 else float(negative_slope) * value for value in _values(values)]


def leaky_relu_derivative(values: Sequence[float], *, negative_slope: float = 0.01) -> list[float]:
    """Return the elementwise derivative of leaky ReLU."""
    _validate_slope(negative_slope)
    return [1.0 if value >= 0 else float(negative_slope) for value in _values(values)]


def sigmoid(values: Sequence[float]) -> list[float]:
    """Return numerically stable logistic probabilities."""
    result = []
    for value in _values(values):
        if value >= 0:
            result.append(1.0 / (1.0 + math.exp(-value)))
        else:
            exponent = math.exp(value)
            result.append(exponent / (1.0 + exponent))
    return result


def silu(values: Sequence[float]) -> list[float]:
    """Return SiLU (also called swish), x * sigmoid(x)."""
    raw = _values(values)
    probabilities = sigmoid(raw)
    return [value * probability for value, probability in zip(raw, probabilities)]


def silu_derivative(values: Sequence[float]) -> list[float]:
    """Return the analytical derivative of SiLU."""
    raw = _values(values)
    probabilities = sigmoid(raw)
    return [probability * (1.0 + value * (1.0 - probability)) for value, probability in zip(raw, probabilities)]


def gelu(values: Sequence[float], *, approximate: bool = True) -> list[float]:
    """Return Gaussian Error Linear Units using exact or tanh approximation."""
    if not isinstance(approximate, bool):
        raise TypeError("approximate must be a boolean")
    result = []
    for value in _values(values):
        if approximate:
            factor = math.sqrt(2.0 / math.pi)
            result.append(0.5 * value * (1.0 + math.tanh(factor * (value + 0.044715 * value**3))))
        else:
            result.append(0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0))))
    return result


def mish(values: Sequence[float], *, beta: float = 1.0, threshold: float = 20.0) -> list[float]:
    """Return the smooth Mish activation, ``x * tanh(softplus(x))``.

    The stable softplus implementation avoids overflow for large positive
    inputs while preserving the negative tail that distinguishes Mish from
    ReLU-like activations.
    """
    raw = _values(values)
    smooth = softplus(raw, beta=beta, threshold=threshold)
    return [value * math.tanh(part) for value, part in zip(raw, smooth)]


def mish_derivative(values: Sequence[float], *, beta: float = 1.0, threshold: float = 20.0) -> list[float]:
    """Return the analytical derivative of Mish for finite numeric inputs."""
    raw = _values(values)
    smooth = softplus(raw, beta=beta, threshold=threshold)
    probabilities = sigmoid([beta * value for value in raw])
    result = []
    for value, part, probability in zip(raw, smooth, probabilities):
        hyperbolic = math.tanh(part)
        result.append(hyperbolic + value * probability * (1.0 - hyperbolic**2))
    return result


def softplus(values: Sequence[float], *, beta: float = 1.0, threshold: float = 20.0) -> list[float]:
    """Return stable ``log(1 + exp(beta*x)) / beta`` values."""
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    result = []
    for value in _values(values):
        scaled = float(beta) * value
        result.append(value if scaled > threshold else math.log1p(math.exp(scaled)) / float(beta))
    return result


def _validate_slope(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("negative_slope must be finite and non-negative")


__all__ = ["gelu", "leaky_relu", "leaky_relu_derivative", "mish", "mish_derivative", "relu", "relu_derivative", "sigmoid", "silu", "silu_derivative", "softplus"]
