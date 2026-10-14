"""Small, dependency-free scaled dot-product attention helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _matrix(values: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of rows")
    rows = [list(row) for row in values]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{name} must be a non-empty rectangular matrix")
    for row in rows:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must contain numbers")
            if not math.isfinite(value):
                raise ValueError(f"{name} must contain finite values")
    return [[float(value) for value in row] for row in rows]


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def scaled_dot_product_attention(
    queries: Sequence[Sequence[float]],
    keys: Sequence[Sequence[float]],
    values: Sequence[Sequence[float]],
    *,
    mask: Sequence[Sequence[bool]] | None = None,
    scale: float | None = None,
) -> tuple[list[list[float]], list[list[float]]]:
    """Return ``(output, weights)`` for scaled dot-product attention.

    ``mask[row][column]`` is ``True`` when a key may be attended to. Fully
    masked rows are returned as zero weights and zero output instead of NaNs.
    """
    query_rows = _matrix(queries, "queries")
    key_rows = _matrix(keys, "keys")
    value_rows = _matrix(values, "values")
    query_width = len(query_rows[0])
    if len(key_rows[0]) != query_width:
        raise ValueError("queries and keys must have the same feature dimension")
    if len(key_rows) != len(value_rows):
        raise ValueError("keys and values must have the same sequence length")
    value_width = len(value_rows[0])
    if mask is not None:
        mask_rows = [list(row) for row in mask]
        if len(mask_rows) != len(query_rows) or any(len(row) != len(key_rows) for row in mask_rows):
            raise ValueError("mask must match query and key sequence lengths")
        if any(not isinstance(item, bool) for row in mask_rows for item in row):
            raise TypeError("mask must contain booleans")
    else:
        mask_rows = [[True] * len(key_rows) for _ in query_rows]
    divisor = math.sqrt(query_width) if scale is None else scale
    if isinstance(divisor, bool) or not isinstance(divisor, (int, float)) or not math.isfinite(divisor) or divisor <= 0:
        raise ValueError("scale must be positive and finite")
    outputs, weights = [], []
    for query, allowed in zip(query_rows, mask_rows):
        scores = [sum(left * right for left, right in zip(query, key)) / divisor for key in key_rows]
        permitted = [score for score, is_allowed in zip(scores, allowed) if is_allowed]
        row_weights = [0.0] * len(key_rows)
        if permitted:
            normalized = _softmax(permitted)
            position = 0
            for index, is_allowed in enumerate(allowed):
                if is_allowed:
                    row_weights[index] = normalized[position]
                    position += 1
        weights.append(row_weights)
        outputs.append([sum(weight * value[index] for weight, value in zip(row_weights, value_rows)) for index in range(value_width)])
    return outputs, weights


def causal_attention_mask(query_length: int, key_length: int | None = None) -> list[list[bool]]:
    """Return a left-to-right causal mask for autoregressive attention."""
    if isinstance(query_length, bool) or not isinstance(query_length, int) or query_length < 1:
        raise ValueError("query_length must be a positive integer")
    if key_length is None:
        key_length = query_length
    if isinstance(key_length, bool) or not isinstance(key_length, int) or key_length < 1:
        raise ValueError("key_length must be a positive integer")
    offset = key_length - query_length
    return [[column <= row + offset for column in range(key_length)] for row in range(query_length)]


def attention_pool(
    queries: Sequence[Sequence[float]], keys: Sequence[Sequence[float]], values: Sequence[Sequence[float]]
) -> list[float]:
    """Pool values using the attention distribution for the first query."""
    output, _ = scaled_dot_product_attention(queries, keys, values)
    return output[0]


__all__ = ["attention_pool", "causal_attention_mask", "scaled_dot_product_attention"]
