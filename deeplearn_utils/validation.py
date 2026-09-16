"""Cross-validation index generators for deep-learning experiments."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any, Iterator


def _validate_count(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def kfold_indices(
    size: int, folds: int = 5, *, shuffle: bool = False, seed: int | None = None
) -> Iterator[tuple[list[int], list[int]]]:
    """Yield train/validation indices for ordinary K-fold cross-validation.

    Every sample appears in exactly one validation fold. Fold sizes differ by
    at most one, with earlier folds receiving the extra samples.
    """
    _validate_count(size, "size", 0)
    _validate_count(folds, "folds", 2)
    if folds > size and size:
        raise ValueError("folds must not exceed the number of samples")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
    indices = list(range(size))
    if shuffle:
        random.Random(seed).shuffle(indices)
    base, remainder = divmod(size, folds)
    cursor = 0
    for fold in range(folds):
        width = base + (fold < remainder)
        validation = indices[cursor : cursor + width]
        validation_set = set(validation)
        train = [index for index in indices if index not in validation_set]
        yield train, validation
        cursor += width


def stratified_kfold_indices(
    labels: Sequence[Any], folds: int = 5, *, shuffle: bool = False, seed: int | None = None
) -> Iterator[tuple[list[int], list[int]]]:
    """Yield K-fold splits with class counts distributed as evenly as possible.

    Each class is partitioned independently before folds are assembled. This
    avoids dropping singleton classes: a singleton is placed in one validation
    fold and remains in training for all other folds. Labels must be hashable.
    """
    if isinstance(labels, (str, bytes)):
        raise TypeError("labels must be a non-string sequence")
    try:
        values = list(labels)
    except TypeError as exc:
        raise TypeError("labels must be a sequence") from exc
    if not values:
        raise ValueError("labels must not be empty")
    _validate_count(folds, "folds", 2)
    if folds > len(values):
        raise ValueError("folds must not exceed the number of samples")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")

    grouped: dict[Any, list[int]] = {}
    for index, label in enumerate(values):
        try:
            grouped.setdefault(label, []).append(index)
        except TypeError as exc:
            raise TypeError("labels must contain hashable values") from exc
    rng = random.Random(seed)
    fold_validation: list[list[int]] = [[] for _ in range(folds)]
    for class_indices in grouped.values():
        ordered = class_indices.copy()
        if shuffle:
            rng.shuffle(ordered)
        for offset, index in enumerate(ordered):
            fold_validation[offset % folds].append(index)
    for validation in fold_validation:
        if shuffle:
            rng.shuffle(validation)
        validation_set = set(validation)
        train = [index for index in range(len(values)) if index not in validation_set]
        yield train, validation


def fold_class_counts(
    labels: Sequence[Any], validation_indices: Sequence[int]
) -> dict[Any, int]:
    """Count labels represented by one validation fold."""
    if isinstance(labels, (str, bytes)) or isinstance(validation_indices, (str, bytes)):
        raise TypeError("labels and validation_indices must be sequences")
    values = list(labels)
    if not values:
        raise ValueError("labels must not be empty")
    counts: dict[Any, int] = {}
    for index in validation_indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("validation indices must be integers")
        if index < 0 or index >= len(values):
            raise ValueError("validation index is outside the label range")
        try:
            counts[values[index]] = counts.get(values[index], 0) + 1
        except TypeError as exc:
            raise TypeError("labels must contain hashable values") from exc
    return counts


__all__ = ["fold_class_counts", "kfold_indices", "stratified_kfold_indices"]
