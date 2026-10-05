"""Sequence preparation helpers for recurrent and transformer models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def sliding_windows(
    sequence: Sequence[Any], window_size: int, *, stride: int = 1, horizon: int = 0
) -> list[tuple[list[Any], list[Any]]]:
    """Create input windows and optional future targets without leakage."""
    if isinstance(sequence, (str, bytes)):
        raise TypeError("sequence must be a non-string sequence")
    values = list(sequence)
    if not values:
        raise ValueError("sequence must not be empty")
    for name, value in (("window_size", window_size), ("stride", stride), ("horizon", horizon)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if window_size < 1 or stride < 1 or horizon < 0:
        raise ValueError("window_size and stride must be positive; horizon cannot be negative")
    result = []
    last_start = len(values) - window_size - horizon
    for start in range(0, max(0, last_start + 1), stride):
        stop = start + window_size
        target_stop = stop + horizon
        result.append((values[start:stop], values[stop:target_stop]))
    if not result:
        raise ValueError("sequence is shorter than window_size plus horizon")
    return result


def temporal_split(
    sequence: Sequence[Any], validation_fraction: float = 0.2, *, test_fraction: float = 0.0
) -> tuple[list[Any], list[Any], list[Any]]:
    """Split ordered data into train, validation, and test segments."""
    if isinstance(sequence, (str, bytes)):
        raise TypeError("sequence must be a non-string sequence")
    values = list(sequence)
    if not values:
        raise ValueError("sequence must not be empty")
    for name, fraction in (("validation_fraction", validation_fraction), ("test_fraction", test_fraction)):
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not 0 <= fraction < 1:
            raise ValueError(f"{name} must be between zero and one")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must leave training data")
    test_size = round(len(values) * test_fraction)
    validation_size = round(len(values) * validation_fraction)
    train_end = len(values) - validation_size - test_size
    validation_end = train_end + validation_size
    return values[:train_end], values[train_end:validation_end], values[validation_end:]


def pad_sequences(
    sequences: Sequence[Sequence[Any]], *, max_length: int | None = None, padding: str = "post", value: Any = 0
) -> tuple[list[list[Any]], list[int]]:
    """Pad variable-length sequences and return padded rows plus original lengths."""
    if isinstance(sequences, (str, bytes)):
        raise TypeError("sequences must be a sequence of sequences")
    rows = [list(row) for row in sequences]
    if not rows:
        raise ValueError("sequences must not be empty")
    if padding not in {"pre", "post"}:
        raise ValueError("padding must be 'pre' or 'post'")
    lengths = [len(row) for row in rows]
    limit = max(lengths) if max_length is None else max_length
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("max_length must be an integer or None")
    if limit < 1:
        raise ValueError("max_length must be positive")
    result = []
    for row in rows:
        clipped = row[-limit:] if padding == "pre" else row[:limit]
        amount = limit - len(clipped)
        filler = [value] * amount
        result.append(filler + clipped if padding == "pre" else clipped + filler)
    return result, [min(length, limit) for length in lengths]


def causal_mask(length: int, *, include_current: bool = True) -> list[list[bool]]:
    """Return a lower-triangular attention mask for autoregressive models."""
    if isinstance(length, bool) or not isinstance(length, int):
        raise TypeError("length must be an integer")
    if length < 1:
        raise ValueError("length must be positive")
    return [[column <= row if include_current else column < row for column in range(length)] for row in range(length)]


__all__ = ["causal_mask", "pad_sequences", "sliding_windows", "temporal_split"]
