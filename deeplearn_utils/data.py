"""Dataset utilities for repeatable deep-learning experiments."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any


def stratified_split(
    labels: Sequence[Any],
    validation_fraction: float = 0.2,
    *,
    seed: int | None = None,
) -> tuple[list[int], list[int]]:
    """Return train and validation indices while preserving class membership.

    Each class is shuffled independently, so the result is reproducible when
    ``seed`` is supplied. Classes with at least two samples receive at least
    one validation item and retain at least one training item; singleton
    classes remain in training because they cannot be split safely.
    """
    if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
        raise TypeError("labels must be a non-string sequence")
    if not labels:
        raise ValueError("labels must not be empty")
    if isinstance(validation_fraction, bool) or not isinstance(
        validation_fraction, (int, float)
    ):
        raise TypeError("validation_fraction must be a number")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")

    by_class: dict[Any, list[int]] = {}
    for index, label in enumerate(labels):
        try:
            by_class.setdefault(label, []).append(index)
        except TypeError as exc:
            raise TypeError("labels must contain hashable values") from exc

    rng = random.Random(seed)
    train: list[int] = []
    validation: list[int] = []
    for indices in by_class.values():
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            train.extend(shuffled)
            continue
        validation_size = max(1, round(len(shuffled) * validation_fraction))
        validation_size = min(validation_size, len(shuffled) - 1)
        validation.extend(shuffled[:validation_size])
        train.extend(shuffled[validation_size:])

    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation
