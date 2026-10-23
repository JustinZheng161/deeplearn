"""Deterministic and seeded sampling helpers for model logits."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def _logits(values: Sequence[float]) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise TypeError("logits must be a numeric sequence")
    result = list(values)
    if not result:
        raise ValueError("logits must not be empty")
    for value in result:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("logits must contain numbers")
        if not math.isfinite(value):
            raise ValueError("logits must contain finite values")
    return [float(value) for value in result]


def _probabilities(logits: Sequence[float], temperature: float) -> list[float]:
    values = _logits(logits)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    scaled = [value / float(temperature) for value in values]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def categorical_sample(probabilities: Sequence[float], *, seed: int | None = None) -> int:
    """Sample one class index from a normalized probability vector."""
    values = list(probabilities)
    if not values or any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in values):
        raise ValueError("probabilities must be non-negative numbers")
    total = sum(values)
    if not math.isclose(total, 1.0, abs_tol=1e-8):
        raise ValueError("probabilities must sum to one")
    generator = _rng(seed)
    target = generator.random()
    cumulative = 0.0
    for index, probability in enumerate(values):
        cumulative += float(probability)
        if target < cumulative or index == len(values) - 1:
            return index
    raise AssertionError("unreachable")


def sample_logits(logits: Sequence[float], *, temperature: float = 1.0, seed: int | None = None) -> int:
    """Sample a class from logits after temperature scaling."""
    return categorical_sample(_probabilities(logits, temperature), seed=seed)


def top_k_filter(logits: Sequence[float], k: int) -> list[float]:
    """Keep the largest ``k`` logits and set the rest to negative infinity."""
    values = _logits(logits)
    _validate_k(k, len(values))
    selected = set(sorted(range(len(values)), key=lambda index: values[index], reverse=True)[:k])
    return [value if index in selected else float("-inf") for index, value in enumerate(values)]


def nucleus_filter(logits: Sequence[float], probability: float = 0.9, *, temperature: float = 1.0) -> list[float]:
    """Keep the smallest probability-sorted set whose mass reaches ``probability``."""
    values = _logits(logits)
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 < probability <= 1:
        raise ValueError("probability must be in (0, 1]")
    probabilities = _probabilities(values, temperature)
    ordered = sorted(range(len(values)), key=lambda index: probabilities[index], reverse=True)
    selected, cumulative = set(), 0.0
    for index in ordered:
        selected.add(index)
        cumulative += probabilities[index]
        if cumulative >= probability:
            break
    return [value if index in selected else float("-inf") for index, value in enumerate(values)]


def top_p_sample(
    logits: Sequence[float], probability: float = 0.9, *, temperature: float = 1.0, seed: int | None = None
) -> int:
    """Sample from a nucleus-filtered, temperature-scaled distribution."""
    filtered = nucleus_filter(logits, probability, temperature=temperature)
    safe = [value if math.isfinite(value) else -1e300 for value in filtered]
    return sample_logits(safe, temperature=temperature, seed=seed)


def _rng(seed: int | None) -> random.Random:
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
    return random.Random(seed)


def _validate_k(k: int, size: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= size:
        raise ValueError("k must be an integer between one and the number of logits")


__all__ = ["categorical_sample", "nucleus_filter", "sample_logits", "top_k_filter", "top_p_sample"]
