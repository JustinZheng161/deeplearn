"""Dependency-free positional encoding helpers for sequence models."""

from __future__ import annotations

import math
from collections.abc import Sequence


def sinusoidal_encoding(length: int, dimension: int, *, offset: int = 0) -> list[list[float]]:
    """Create the deterministic sine/cosine encoding from Transformer models."""
    _positive_integer(length, "length")
    _positive_integer(dimension, "dimension")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    result = []
    for position in range(offset, offset + length):
        row = []
        for index in range(dimension):
            exponent = 2 * (index // 2) / dimension
            angle = position / (10000.0**exponent)
            row.append(math.sin(angle) if index % 2 == 0 else math.cos(angle))
        result.append(row)
    return result


def learned_position_lookup(
    positions: Sequence[int], table: Sequence[Sequence[float]]
) -> list[list[float]]:
    """Gather position vectors from a learned embedding table."""
    rows = [list(row) for row in table]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("table must be a non-empty rectangular matrix")
    result = []
    for position in positions:
        if isinstance(position, bool) or not isinstance(position, int):
            raise TypeError("positions must contain integers")
        if not 0 <= position < len(rows):
            raise ValueError("position is outside the embedding table")
        result.append([float(value) for value in rows[position]])
    return result


def rotary_frequencies(dimension: int, *, base: float = 10000.0) -> list[float]:
    """Return inverse frequencies for even-dimensional rotary embeddings."""
    _positive_integer(dimension, "dimension")
    if dimension % 2:
        raise ValueError("dimension must be even for rotary frequencies")
    if isinstance(base, bool) or not isinstance(base, (int, float)) or not math.isfinite(base) or base <= 1:
        raise ValueError("base must be greater than one and finite")
    return [1.0 / (float(base) ** (index / dimension)) for index in range(0, dimension, 2)]


def rotate_pair(vector: Sequence[float], position: int, *, base: float = 10000.0) -> list[float]:
    """Apply rotary position embedding to one even-dimensional vector."""
    values = list(vector)
    if not values or len(values) % 2 or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError("vector must contain finite values and have even dimension")
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise ValueError("position must be a non-negative integer")
    frequencies = rotary_frequencies(len(values), base=base)
    result = []
    for index, frequency in enumerate(frequencies):
        cosine = math.cos(position * frequency)
        sine = math.sin(position * frequency)
        first, second = float(values[2 * index]), float(values[2 * index + 1])
        result.extend([first * cosine - second * sine, first * sine + second * cosine])
    return result


def add_position_encoding(
    values: Sequence[Sequence[float]], encoding: Sequence[Sequence[float]]
) -> list[list[float]]:
    """Add position vectors to a sequence of hidden-state vectors."""
    hidden = [list(row) for row in values]
    positions = [list(row) for row in encoding]
    if not hidden or len(hidden) != len(positions) or any(len(left) != len(right) for left, right in zip(hidden, positions)):
        raise ValueError("values and encoding must have the same non-empty shape")
    return [[float(left) + float(right) for left, right in zip(hidden_row, position_row)] for hidden_row, position_row in zip(hidden, positions)]


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


__all__ = ["add_position_encoding", "learned_position_lookup", "rotate_pair", "rotary_frequencies", "sinusoidal_encoding"]
