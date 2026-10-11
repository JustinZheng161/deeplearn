"""Small, framework-agnostic helpers for training loops.

These utilities deliberately operate on Python sequences and callbacks. They
can be used around PyTorch, TensorFlow, or custom model code without forcing a
specific framework into this dependency-light package.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar


T = TypeVar("T")
U = TypeVar("U")


def batch_indices(
    size: int, batch_size: int, *, shuffle: bool = False, seed: int | None = None
) -> Iterator[list[int]]:
    """Yield index batches covering ``range(size)`` exactly once.

    The final batch may be smaller than ``batch_size``. A local random number
    generator is used, so shuffling does not alter the caller's global RNG.
    """
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size < 0:
        raise ValueError("size must be non-negative")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")

    indices = list(range(size))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, size, batch_size):
        yield indices[start : start + batch_size]


def iter_minibatches(
    features: Sequence[T],
    targets: Sequence[U],
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int | None = None,
) -> Iterator[tuple[list[T], list[U]]]:
    """Yield aligned feature and target mini-batches.

    Both outputs are new lists, so a training step may safely mutate a batch
    without changing the source dataset. Reproducible shuffling is available
    through ``seed``.
    """
    if isinstance(features, (str, bytes)) or isinstance(targets, (str, bytes)):
        raise TypeError("features and targets must be sequences")
    if len(features) != len(targets):
        raise ValueError("features and targets must have the same length")
    for indices in batch_indices(len(features), batch_size, shuffle=shuffle, seed=seed):
        yield ([features[index] for index in indices], [targets[index] for index in indices])


def clip_by_global_norm(
    gradients: Sequence[Sequence[float]], max_norm: float
) -> tuple[list[list[float]], float]:
    """Clip gradient rows by one global L2 norm and return original norm.

    The returned gradients preserve shape and are scaled uniformly when the
    global norm exceeds ``max_norm``. Empty rows are allowed, but the outer
    sequence must contain at least one gradient vector.
    """
    if isinstance(gradients, (str, bytes)):
        raise TypeError("gradients must be a sequence of vectors")
    try:
        rows = [list(row) for row in gradients]
    except TypeError as exc:
        raise TypeError("gradients must be a sequence of vectors") from exc
    if not rows:
        raise ValueError("gradients must not be empty")
    if isinstance(max_norm, bool) or not isinstance(max_norm, (int, float)):
        raise TypeError("max_norm must be a number")
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    squared = 0.0
    for row in rows:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("gradients must contain numbers")
            squared += float(value) ** 2
    global_norm = squared ** 0.5
    if global_norm == 0 or global_norm <= max_norm:
        return [[float(value) for value in row] for row in rows], global_norm
    scale = float(max_norm) / global_norm
    return [[float(value) * scale for value in row] for row in rows], global_norm


@dataclass
class RunningAverage:
    """Track a weighted mean such as loss over batches of unequal size."""

    total: float = 0.0
    weight: int = 0

    def update(self, value: float, *, weight: int = 1) -> None:
        """Add ``value`` with the supplied positive sample weight."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("value must be a number")
        if isinstance(weight, bool) or not isinstance(weight, int):
            raise TypeError("weight must be an integer")
        if weight < 1:
            raise ValueError("weight must be positive")
        self.total += float(value) * weight
        self.weight += weight

    @property
    def mean(self) -> float:
        """Return the current mean, or zero before the first update."""
        return self.total / self.weight if self.weight else 0.0

    def reset(self) -> None:
        """Clear accumulated values for a new epoch."""
        self.total = 0.0
        self.weight = 0


@dataclass
class ParameterEMA:
    """Track an exponential moving average of flat model parameters."""

    decay: float = 0.999
    shadow: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.decay, bool) or not isinstance(self.decay, (int, float)):
            raise TypeError("decay must be a number")
        if not 0 <= self.decay < 1:
            raise ValueError("decay must be between zero and one")

    def update(self, parameters: Sequence[float]) -> list[float]:
        """Update the average and return a copy of the smoothed parameters."""
        values = list(parameters)
        if not values:
            raise ValueError("parameters must not be empty")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise TypeError("parameters must contain numbers")
        if not self.shadow:
            self.shadow = [float(value) for value in values]
        elif len(self.shadow) != len(values):
            raise ValueError("parameter dimension changed after initialization")
        else:
            self.shadow = [self.decay * old + (1 - self.decay) * float(value) for old, value in zip(self.shadow, values)]
        return self.shadow.copy()

    def reset(self) -> None:
        """Discard the running parameter average."""
        self.shadow.clear()


@dataclass(frozen=True)
class EarlyStoppingState:
    """Serializable snapshot of early-stopping progress."""

    best: float | None
    bad_epochs: int
    stopped: bool


class EarlyStopping:
    """Stop training when a monitored validation value stops improving."""

    def __init__(
        self,
        patience: int = 5,
        *,
        mode: str = "min",
        min_delta: float = 0.0,
        restore_best: bool = True,
    ) -> None:
        if isinstance(patience, bool) or not isinstance(patience, int):
            raise TypeError("patience must be an integer")
        if patience < 1:
            raise ValueError("patience must be positive")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if isinstance(min_delta, bool) or not isinstance(min_delta, (int, float)):
            raise TypeError("min_delta must be a number")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        self.patience = patience
        self.mode = mode
        self.min_delta = float(min_delta)
        self.restore_best = restore_best
        self.best: float | None = None
        self.bad_epochs = 0
        self.stopped = False

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def update(self, value: float) -> bool:
        """Record an epoch value and return whether training should stop."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("value must be a number")
        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError("value must be a number")
        if self._improved(float(value)):
            self.best = float(value)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        self.stopped = self.bad_epochs >= self.patience
        return self.stopped

    def state(self) -> EarlyStoppingState:
        """Return a compact snapshot suitable for checkpoint metadata."""
        return EarlyStoppingState(self.best, self.bad_epochs, self.stopped)

    def reset(self) -> None:
        """Start monitoring a new training run."""
        self.best = None
        self.bad_epochs = 0
        self.stopped = False


__all__ = [
    "EarlyStopping",
    "EarlyStoppingState",
    "ParameterEMA",
    "clip_by_global_norm",
    "RunningAverage",
    "batch_indices",
    "iter_minibatches",
]
