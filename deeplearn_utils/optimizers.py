"""Small framework-agnostic optimizer updates for educational experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections.abc import Sequence


def _vector(values: Sequence[float], name: str) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence")
    result = list(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    for value in result:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must contain numbers")
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain finite values")
    return [float(value) for value in result]


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


@dataclass
class SGD:
    """Stateful SGD with optional momentum, dampening, and Nesterov updates."""

    learning_rate: float = 1e-2
    momentum: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False
    velocity: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        _positive(self.learning_rate, "learning_rate")
        if not isinstance(self.momentum, (int, float)) or isinstance(self.momentum, bool) or not 0 <= self.momentum < 1:
            raise ValueError("momentum must be between zero and one")
        if not isinstance(self.weight_decay, (int, float)) or isinstance(self.weight_decay, bool) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not isinstance(self.nesterov, bool):
            raise TypeError("nesterov must be a boolean")
        if self.nesterov and self.momentum == 0:
            raise ValueError("nesterov requires positive momentum")

    def step(self, parameters: Sequence[float], gradients: Sequence[float]) -> list[float]:
        """Return parameters after one SGD update."""
        values = _vector(parameters, "parameters")
        updates = _vector(gradients, "gradients")
        if len(values) != len(updates):
            raise ValueError("parameters and gradients must have the same length")
        if self.velocity and len(self.velocity) != len(values):
            raise ValueError("parameter dimension changed after optimizer initialization")
        if not self.velocity:
            self.velocity = [0.0] * len(values)
        adjusted = [gradient + float(self.weight_decay) * value for value, gradient in zip(values, updates)]
        if self.momentum:
            self.velocity = [self.momentum * old + gradient for old, gradient in zip(self.velocity, adjusted)]
            direction = [gradient + self.momentum * velocity for gradient, velocity in zip(adjusted, self.velocity)] if self.nesterov else self.velocity
        else:
            direction = adjusted
        return [value - self.learning_rate * gradient for value, gradient in zip(values, direction)]

    def reset(self) -> None:
        """Clear momentum state."""
        self.velocity.clear()


@dataclass
class Adam:
    """Adam optimizer with optional decoupled weight decay."""

    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    first_moment: list[float] = field(default_factory=list)
    second_moment: list[float] = field(default_factory=list)
    step_count: int = 0

    def __post_init__(self) -> None:
        _positive(self.learning_rate, "learning_rate")
        for name, value in (("beta1", self.beta1), ("beta2", self.beta2)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
                raise ValueError(f"{name} must be between zero and one")
        _positive(self.epsilon, "epsilon")
        if isinstance(self.weight_decay, bool) or not isinstance(self.weight_decay, (int, float)) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

    def step(self, parameters: Sequence[float], gradients: Sequence[float]) -> list[float]:
        """Return parameters after one bias-corrected Adam update."""
        values = _vector(parameters, "parameters")
        updates = _vector(gradients, "gradients")
        if len(values) != len(updates):
            raise ValueError("parameters and gradients must have the same length")
        if self.first_moment and len(self.first_moment) != len(values):
            raise ValueError("parameter dimension changed after optimizer initialization")
        if not self.first_moment:
            self.first_moment = [0.0] * len(values)
            self.second_moment = [0.0] * len(values)
        self.step_count += 1
        self.first_moment = [self.beta1 * old + (1 - self.beta1) * gradient for old, gradient in zip(self.first_moment, updates)]
        self.second_moment = [self.beta2 * old + (1 - self.beta2) * gradient * gradient for old, gradient in zip(self.second_moment, updates)]
        correction1 = 1 - self.beta1**self.step_count
        correction2 = 1 - self.beta2**self.step_count
        result = []
        for value, first, second in zip(values, self.first_moment, self.second_moment):
            mean = first / correction1
            variance = second / correction2
            decayed = value * (1 - self.learning_rate * self.weight_decay)
            result.append(decayed - self.learning_rate * mean / (math.sqrt(variance) + self.epsilon))
        return result

    def reset(self) -> None:
        """Reset moments and step counter."""
        self.first_moment.clear()
        self.second_moment.clear()
        self.step_count = 0


__all__ = ["Adam", "SGD"]
