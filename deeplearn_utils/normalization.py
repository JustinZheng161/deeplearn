"""Framework-agnostic normalization helpers for small neural-network experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field


def _matrix(values: Sequence[Sequence[float]]) -> list[list[float]]:
    if isinstance(values, (str, bytes)):
        raise TypeError("values must be a sequence of rows")
    rows = [list(row) for row in values]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("values must be a non-empty rectangular matrix")
    for row in rows:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("values must contain numbers")
            if not math.isfinite(value):
                raise ValueError("values must contain finite numbers")
    return [[float(value) for value in row] for row in rows]


def _epsilon(epsilon: float) -> None:
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")


def layer_norm(
    values: Sequence[Sequence[float]], *, gamma: Sequence[float] | None = None, beta: Sequence[float] | None = None, epsilon: float = 1e-5
) -> list[list[float]]:
    """Normalize each row independently across its feature dimension."""
    rows = _matrix(values)
    _epsilon(epsilon)
    width = len(rows[0])
    scale = _features(gamma, width, "gamma", 1.0)
    shift = _features(beta, width, "beta", 0.0)
    result = []
    for row in rows:
        mean = sum(row) / width
        variance = sum((value - mean) ** 2 for value in row) / width
        denominator = math.sqrt(variance + epsilon)
        result.append([(value - mean) / denominator * weight + bias for value, weight, bias in zip(row, scale, shift)])
    return result


def rms_norm(
    values: Sequence[Sequence[float]], *, gamma: Sequence[float] | None = None, epsilon: float = 1e-5
) -> list[list[float]]:
    """Normalize each row by its root-mean-square without centering."""
    rows = _matrix(values)
    _epsilon(epsilon)
    scale = _features(gamma, len(rows[0]), "gamma", 1.0)
    return [[value / math.sqrt(sum(item * item for item in row) / len(row) + epsilon) * weight for value, weight in zip(row, scale)] for row in rows]


def batch_norm(
    values: Sequence[Sequence[float]], *, gamma: Sequence[float] | None = None, beta: Sequence[float] | None = None, epsilon: float = 1e-5
) -> list[list[float]]:
    """Normalize each feature column across the batch."""
    rows = _matrix(values)
    _epsilon(epsilon)
    width = len(rows[0])
    scale = _features(gamma, width, "gamma", 1.0)
    shift = _features(beta, width, "beta", 0.0)
    means = [sum(row[column] for row in rows) / len(rows) for column in range(width)]
    variances = [sum((row[column] - means[column]) ** 2 for row in rows) / len(rows) for column in range(width)]
    return [[(row[column] - means[column]) / math.sqrt(variances[column] + epsilon) * scale[column] + shift[column] for column in range(width)] for row in rows]


@dataclass
class RunningBatchNorm:
    """Track running feature statistics and normalize future batches."""

    features: int
    momentum: float = 0.1
    epsilon: float = 1e-5
    running_mean: list[float] = field(default_factory=list)
    running_variance: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.features, bool) or not isinstance(self.features, int) or self.features < 1:
            raise ValueError("features must be a positive integer")
        if isinstance(self.momentum, bool) or not isinstance(self.momentum, (int, float)) or not 0 < self.momentum <= 1:
            raise ValueError("momentum must be between zero and one")
        _epsilon(self.epsilon)
        if not self.running_mean:
            self.running_mean = [0.0] * self.features
            self.running_variance = [1.0] * self.features

    def update(self, values: Sequence[Sequence[float]]) -> None:
        """Update running statistics from one batch."""
        rows = _matrix(values)
        if len(rows[0]) != self.features:
            raise ValueError("batch feature count does not match the running statistics")
        means = [sum(row[column] for row in rows) / len(rows) for column in range(self.features)]
        variances = [sum((row[column] - means[column]) ** 2 for row in rows) / len(rows) for column in range(self.features)]
        self.running_mean = [(1 - self.momentum) * old + self.momentum * current for old, current in zip(self.running_mean, means)]
        self.running_variance = [(1 - self.momentum) * old + self.momentum * current for old, current in zip(self.running_variance, variances)]

    def transform(self, values: Sequence[Sequence[float]]) -> list[list[float]]:
        """Normalize rows using the tracked running statistics."""
        rows = _matrix(values)
        if len(rows[0]) != self.features:
            raise ValueError("batch feature count does not match the running statistics")
        return [[(value - self.running_mean[column]) / math.sqrt(self.running_variance[column] + self.epsilon) for column, value in enumerate(row)] for row in rows]


def _features(values: Sequence[float] | None, width: int, name: str, default: float) -> list[float]:
    if values is None:
        return [default] * width
    result = list(values)
    if len(result) != width:
        raise ValueError(f"{name} must match the feature count")
    for value in result:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must contain finite numbers")
    return [float(value) for value in result]


__all__ = ["RunningBatchNorm", "batch_norm", "layer_norm", "rms_norm"]
