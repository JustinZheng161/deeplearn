"""Losses and diagnostics for sparse probability distributions."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _probabilities(values: Sequence[float]) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise TypeError("probabilities must be a numeric sequence")
    result = list(values)
    if not result:
        raise ValueError("probabilities must not be empty")
    for value in result:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("probabilities must be finite and non-negative")
    if not math.isclose(sum(result), 1.0, abs_tol=1e-8):
        raise ValueError("probabilities must sum to one")
    return [float(value) for value in result]


def sparse_cross_entropy(probabilities: Sequence[float], target: int) -> float:
    """Return the negative log likelihood of one target class."""
    values = _probabilities(probabilities)
    if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < len(values):
        raise ValueError("target must be a valid class index")
    return -math.log(max(values[target], 1e-15))


def sparse_entropy(probabilities: Sequence[float]) -> float:
    """Return Shannon entropy, ignoring exact zero probabilities."""
    return -sum(value * math.log(value) for value in _probabilities(probabilities) if value > 0)


def support_size(probabilities: Sequence[float], *, tolerance: float = 1e-8) -> int:
    """Count probabilities whose mass exceeds ``tolerance``."""
    values = _probabilities(probabilities)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not 0 <= tolerance <= 1:
        raise ValueError("tolerance must be between zero and one")
    return sum(value > tolerance for value in values)


def top_probability(probabilities: Sequence[float]) -> tuple[int, float]:
    """Return the first class index and mass of the most likely class."""
    values = _probabilities(probabilities)
    index = max(range(len(values)), key=values.__getitem__)
    return index, values[index]


def top_k_mass(probabilities: Sequence[float], k: int) -> float:
    """Return the probability mass contained in the largest ``k`` classes."""
    values = _probabilities(probabilities)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(values):
        raise ValueError("k must be between one and the number of classes")
    return sum(sorted(values, reverse=True)[:k])


def sparse_brier_score(probabilities: Sequence[float], target: int) -> float:
    """Return the multiclass Brier score for one sparse target label."""
    values = _probabilities(probabilities)
    if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < len(values):
        raise ValueError("target must be a valid class index")
    return sum((value - (1.0 if index == target else 0.0)) ** 2 for index, value in enumerate(values))


__all__ = ["sparse_brier_score", "sparse_cross_entropy", "sparse_entropy", "support_size", "top_k_mass", "top_probability"]
