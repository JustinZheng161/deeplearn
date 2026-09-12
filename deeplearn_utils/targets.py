"""Target preparation helpers for classification training."""

from __future__ import annotations

from collections.abc import Sequence


def label_smoothed_targets(
    labels: Sequence[int], class_count: int, *, smoothing: float = 0.1
) -> list[list[float]]:
    """Convert integer labels into label-smoothed probability distributions.

    The true class receives ``1 - smoothing`` and the remaining classes share
    the smoothing mass. The returned nested lists are framework-agnostic and
    can be converted to tensors by the caller.
    """
    if isinstance(labels, (str, bytes)):
        raise TypeError("labels must be a sequence of integer class indices")
    if isinstance(class_count, bool) or not isinstance(class_count, int):
        raise TypeError("class_count must be an integer")
    if class_count < 2:
        raise ValueError("class_count must be at least two")
    if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)):
        raise TypeError("smoothing must be a number")
    if not 0 <= smoothing < 1:
        raise ValueError("smoothing must be between zero and one")

    try:
        values = list(labels)
    except TypeError as exc:
        raise TypeError("labels must be a sequence of integer class indices") from exc
    if any(isinstance(label, bool) or not isinstance(label, int) for label in values):
        raise TypeError("labels must contain only integer class indices")
    invalid = [label for label in values if label < 0 or label >= class_count]
    if invalid:
        raise ValueError(f"label {invalid[0]} is outside the class range")

    off_value = float(smoothing) / (class_count - 1)
    on_value = 1.0 - float(smoothing)
    return [
        [on_value if class_index == label else off_value for class_index in range(class_count)]
        for label in values
    ]


__all__ = ["label_smoothed_targets"]
