"""Reproducible dropout helpers for lightweight neural-network experiments."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any


def _rate(rate: float) -> None:
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 <= rate < 1:
        raise ValueError("rate must be between zero inclusive and one exclusive")


def _rng(seed: int | None) -> random.Random:
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
    return random.Random(seed)


def dropout(values: Sequence[float], rate: float = 0.5, *, seed: int | None = None, training: bool = True) -> list[float]:
    """Apply elementwise inverted dropout to a numeric vector."""
    _rate(rate)
    if not isinstance(training, bool):
        raise TypeError("training must be a boolean")
    result = _numbers(values)
    if not training or rate == 0:
        return result
    keep_probability = 1.0 - float(rate)
    generator = _rng(seed)
    return [value / keep_probability if generator.random() < keep_probability else 0.0 for value in result]


def feature_dropout(
    values: Sequence[Sequence[float]], rate: float = 0.5, *, seed: int | None = None, training: bool = True
) -> list[list[float]]:
    """Drop entire feature columns consistently across all rows."""
    rows = _matrix(values)
    _rate(rate)
    if not isinstance(training, bool):
        raise TypeError("training must be a boolean")
    if not training or rate == 0:
        return [row.copy() for row in rows]
    keep_probability = 1.0 - float(rate)
    generator = _rng(seed)
    keep = [generator.random() < keep_probability for _ in rows[0]]
    return [[value / keep_probability if keep[column] else 0.0 for column, value in enumerate(row)] for row in rows]


def stochastic_depth(
    residual: Sequence[float], shortcut: Sequence[float], drop_rate: float = 0.0, *, seed: int | None = None, training: bool = True
) -> list[float]:
    """Randomly drop a residual branch while preserving expected scale."""
    branch = _numbers(residual)
    skip = _numbers(shortcut)
    if len(branch) != len(skip):
        raise ValueError("residual and shortcut must have the same length")
    _rate(drop_rate)
    if not training:
        return [left + right for left, right in zip(branch, skip)]
    if drop_rate == 0:
        return [left + right for left, right in zip(branch, skip)]
    generator = _rng(seed)
    if generator.random() >= 1.0 - float(drop_rate):
        return skip
    scale = 1.0 / (1.0 - float(drop_rate))
    return [left * scale + right for left, right in zip(branch, skip)]


def dropout_mask(length: int, rate: float = 0.5, *, seed: int | None = None) -> list[bool]:
    """Generate a deterministic keep mask without applying it."""
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("length must be a positive integer")
    _rate(rate)
    generator = _rng(seed)
    return [generator.random() >= float(rate) for _ in range(length)]


def _numbers(values: Sequence[float]) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise TypeError("values must be a numeric sequence")
    result = list(values)
    if not result:
        raise ValueError("values must not be empty")
    for value in result:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("values must contain numbers")
    return [float(value) for value in result]


def _matrix(values: Sequence[Sequence[float]]) -> list[list[float]]:
    if isinstance(values, (str, bytes)):
        raise TypeError("values must be a sequence of rows")
    rows = [_numbers(row) for row in values]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("values must be a non-empty rectangular matrix")
    return rows


__all__ = ["dropout", "dropout_mask", "feature_dropout", "stochastic_depth"]
