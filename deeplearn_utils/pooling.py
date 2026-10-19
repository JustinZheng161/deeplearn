"""Dependency-free pooling helpers for sequence and feature tensors."""

from __future__ import annotations

import math
from collections.abc import Sequence


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


def _mask(mask: Sequence[bool] | None, length: int) -> list[bool]:
    if mask is None:
        return [True] * length
    values = list(mask)
    if len(values) != length:
        raise ValueError("mask must match the sequence length")
    if any(not isinstance(value, bool) for value in values):
        raise TypeError("mask must contain booleans")
    if not any(values):
        raise ValueError("mask must select at least one element")
    return values


def mean_pool(values: Sequence[Sequence[float]], *, mask: Sequence[bool] | None = None) -> list[float]:
    """Average selected rows feature by feature."""
    rows = _matrix(values)
    selected = [row for row, included in zip(rows, _mask(mask, len(rows))) if included]
    return [sum(row[column] for row in selected) / len(selected) for column in range(len(rows[0]))]


def sum_pool(values: Sequence[Sequence[float]], *, mask: Sequence[bool] | None = None) -> list[float]:
    """Sum selected rows feature by feature."""
    rows = _matrix(values)
    selected = [row for row, included in zip(rows, _mask(mask, len(rows))) if included]
    return [sum(row[column] for row in selected) for column in range(len(rows[0]))]


def max_pool(values: Sequence[Sequence[float]], *, mask: Sequence[bool] | None = None) -> list[float]:
    """Take the maximum selected value feature by feature."""
    rows = _matrix(values)
    selected = [row for row, included in zip(rows, _mask(mask, len(rows))) if included]
    return [max(row[column] for row in selected) for column in range(len(rows[0]))]


def min_pool(values: Sequence[Sequence[float]], *, mask: Sequence[bool] | None = None) -> list[float]:
    """Take the minimum selected value feature by feature."""
    rows = _matrix(values)
    selected = [row for row, included in zip(rows, _mask(mask, len(rows))) if included]
    return [min(row[column] for row in selected) for column in range(len(rows[0]))]


def logsumexp_pool(values: Sequence[Sequence[float]], *, mask: Sequence[bool] | None = None) -> list[float]:
    """Pool with a numerically stable log-sum-exp reduction."""
    rows = _matrix(values)
    selected = [row for row, included in zip(rows, _mask(mask, len(rows))) if included]
    result = []
    for column in range(len(rows[0])):
        maximum = max(row[column] for row in selected)
        result.append(maximum + math.log(sum(math.exp(row[column] - maximum) for row in selected)))
    return result


def masked_softmax(values: Sequence[float], *, mask: Sequence[bool] | None = None) -> list[float]:
    """Return a stable softmax assigning zero probability to masked entries."""
    numbers = list(values)
    if not numbers:
        raise ValueError("values must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in numbers):
        raise ValueError("values must contain finite numbers")
    allowed = _mask(mask, len(numbers))
    selected = [value for value, include in zip(numbers, allowed) if include]
    maximum = max(selected)
    exponentials = [math.exp(value - maximum) if include else 0.0 for value, include in zip(numbers, allowed)]
    total = sum(exponentials)
    return [value / total for value in exponentials]


__all__ = ["logsumexp_pool", "masked_softmax", "max_pool", "mean_pool", "min_pool", "sum_pool"]
