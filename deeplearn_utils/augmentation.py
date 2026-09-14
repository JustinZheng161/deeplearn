"""Dependency-light data augmentation for numeric training batches."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any


def _rows(features: Sequence[Sequence[float]]) -> list[list[float]]:
    if isinstance(features, (str, bytes)):
        raise TypeError("features must be a sequence of rows")
    try:
        rows = [list(row) for row in features]
    except TypeError as exc:
        raise TypeError("features must be a sequence of rows") from exc
    if not rows:
        raise ValueError("features must not be empty")
    width = len(rows[0])
    if not width or any(len(row) != width for row in rows):
        raise ValueError("features must be a non-empty rectangular matrix")
    for row in rows:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("features must contain numeric values")
            if not math.isfinite(value):
                raise ValueError("features must contain finite values")
    return [[float(value) for value in row] for row in rows]


def _probability(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return float(value)


def add_gaussian_noise(
    features: Sequence[Sequence[float]], *, stddev: float = 0.01, seed: int | None = None
) -> list[list[float]]:
    """Add independent zero-mean Gaussian noise without mutating inputs."""
    rows = _rows(features)
    if isinstance(stddev, bool) or not isinstance(stddev, (int, float)):
        raise TypeError("stddev must be a number")
    if not math.isfinite(stddev) or stddev < 0:
        raise ValueError("stddev must be finite and non-negative")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
    rng = random.Random(seed)
    return [[value + rng.gauss(0.0, float(stddev)) for value in row] for row in rows]


def random_feature_dropout(
    features: Sequence[Sequence[float]], *, probability: float = 0.1, seed: int | None = None
) -> list[list[float]]:
    """Set individual feature values to zero with a reproducible probability."""
    rows = _rows(features)
    probability = _probability(probability, "probability")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
    rng = random.Random(seed)
    return [[0.0 if rng.random() < probability else value for value in row] for row in rows]


def mixup(
    features: Sequence[Sequence[float]],
    targets: Sequence[Any],
    *,
    alpha: float = 0.2,
    seed: int | None = None,
) -> tuple[list[list[float]], list[tuple[Any, Any, float]]]:
    """Mix each row with a randomly selected partner.

    Targets are returned as ``(target_a, target_b, weight_a)`` tuples so the
    caller can construct soft labels appropriate to its framework or task.
    The same batch size and feature width are preserved.
    """
    rows = _rows(features)
    if isinstance(targets, (str, bytes)):
        raise TypeError("targets must be a sequence")
    labels = list(targets)
    if len(rows) != len(labels):
        raise ValueError("features and targets must have the same length")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a number")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive and finite")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
    rng = random.Random(seed)
    partners = list(range(len(rows)))
    rng.shuffle(partners)
    mixed: list[list[float]] = []
    mixed_targets: list[tuple[Any, Any, float]] = []
    for index, partner in enumerate(partners):
        weight = rng.betavariate(float(alpha), float(alpha))
        mixed.append([
            weight * left + (1.0 - weight) * right
            for left, right in zip(rows[index], rows[partner])
        ])
        mixed_targets.append((labels[index], labels[partner], weight))
    return mixed, mixed_targets


def compose_augmentations(
    features: Sequence[Sequence[float]],
    *,
    noise_stddev: float = 0.0,
    dropout_probability: float = 0.0,
    seed: int | None = None,
) -> list[list[float]]:
    """Apply noise then feature dropout using independent derived streams."""
    result = _rows(features)
    if noise_stddev < 0:
        raise ValueError("noise_stddev must be non-negative")
    if dropout_probability < 0:
        raise ValueError("dropout_probability must be non-negative")
    if noise_stddev:
        result = add_gaussian_noise(result, stddev=noise_stddev, seed=seed)
    if dropout_probability:
        dropout_seed = None if seed is None else seed + 1
        result = random_feature_dropout(
            result, probability=dropout_probability, seed=dropout_seed
        )
    return result


__all__ = [
    "add_gaussian_noise",
    "compose_augmentations",
    "mixup",
    "random_feature_dropout",
]
