"""Lookahead optimization for the dependency-light optimizer primitives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class Lookahead:
    """Wrap an optimizer with periodic interpolation toward slow weights.

    The wrapped optimizer performs the fast updates. Every ``sync_period``
    calls, the slow weights move toward those fast weights by ``alpha`` and the
    interpolated values are returned. This keeps the wrapper independent of
    the concrete optimizer implementation while preserving its internal state.
    """

    def __init__(self, optimizer: Any, *, alpha: float = 0.5, sync_period: int = 5) -> None:
        if not hasattr(optimizer, "step") or not callable(optimizer.step):
            raise TypeError("optimizer must provide a callable step method")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a number")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be greater than zero and at most one")
        if isinstance(sync_period, bool) or not isinstance(sync_period, int):
            raise TypeError("sync_period must be an integer")
        if sync_period < 1:
            raise ValueError("sync_period must be positive")
        self.optimizer = optimizer
        self.alpha = float(alpha)
        self.sync_period = sync_period
        self.step_count = 0
        self.slow_parameters: list[float] = []

    @staticmethod
    def _values(values: Sequence[float], name: str) -> list[float]:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{name} must be a numeric sequence")
        result = list(values)
        if not result:
            raise ValueError(f"{name} must not be empty")
        for value in result:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must contain numbers")
        return [float(value) for value in result]

    def step(self, parameters: Sequence[float], gradients: Sequence[float]) -> list[float]:
        """Apply one fast update and synchronize slow weights when due."""
        values = self._values(parameters, "parameters")
        updates = self._values(gradients, "gradients")
        if len(values) != len(updates):
            raise ValueError("parameters and gradients must have the same length")
        if self.slow_parameters and len(self.slow_parameters) != len(values):
            raise ValueError("parameter dimension changed after initialization")
        fast_parameters = self._values(self.optimizer.step(values, updates), "optimizer result")
        if len(fast_parameters) != len(values):
            raise ValueError("optimizer result must match parameter dimensions")
        if not self.slow_parameters:
            self.slow_parameters = values.copy()
        self.step_count += 1
        if self.step_count % self.sync_period:
            return fast_parameters
        self.slow_parameters = [
            slow + self.alpha * (fast - slow)
            for slow, fast in zip(self.slow_parameters, fast_parameters)
        ]
        return self.slow_parameters.copy()

    def reset(self) -> None:
        """Reset wrapper state and the wrapped optimizer when supported."""
        self.slow_parameters.clear()
        self.step_count = 0
        reset = getattr(self.optimizer, "reset", None)
        if reset is not None:
            reset()

    @property
    def synchronized(self) -> bool:
        """Whether the most recent call performed a slow-weight sync."""
        return self.step_count > 0 and self.step_count % self.sync_period == 0


__all__ = ["Lookahead"]
