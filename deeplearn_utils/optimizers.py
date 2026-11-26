"""Small framework-agnostic optimizer updates for educational experiments."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field, fields
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


class OptimizerStateMixin:
    """Provide checkpoint-friendly serialization for dataclass optimizers."""

    def state_dict(self) -> dict[str, object]:
        """Return an independent copy of hyperparameters and optimizer state."""
        return {item.name: deepcopy(getattr(self, item.name)) for item in fields(self)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore a state produced by :meth:`state_dict`.

        Loading validates the optimizer configuration before mutating it and
        deep-copies values so callers can safely reuse or discard checkpoints.
        """
        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary")
        expected = {item.name for item in fields(self)}
        if set(state) != expected:
            missing = sorted(expected - set(state))
            extra = sorted(set(state) - expected)
            details = f"missing={missing}" if missing else f"unexpected={extra}"
            raise ValueError(f"state keys do not match optimizer fields ({details})")
        original = self.state_dict()
        candidate = {name: deepcopy(value) for name, value in state.items()}
        for name, value in candidate.items():
            setattr(self, name, value)
        try:
            self.__post_init__()
        except (TypeError, ValueError):
            for name, value in original.items():
                setattr(self, name, value)
            raise


@dataclass
class SGD(OptimizerStateMixin):
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
class Adam(OptimizerStateMixin):
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


@dataclass
class NAdam(OptimizerStateMixin):
    """Adam with a bias-corrected Nesterov first-moment update."""

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
        if isinstance(self.weight_decay, bool) or not isinstance(self.weight_decay, (int, float)):
            raise TypeError("weight_decay must be a number")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative and finite")

    def step(self, parameters: Sequence[float], gradients: Sequence[float]) -> list[float]:
        """Return parameters after one NAdam update."""
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
        self.first_moment = [
            self.beta1 * old + (1 - self.beta1) * gradient
            for old, gradient in zip(self.first_moment, updates)
        ]
        self.second_moment = [
            self.beta2 * old + (1 - self.beta2) * gradient * gradient
            for old, gradient in zip(self.second_moment, updates)
        ]
        first_correction = 1 - self.beta1 ** self.step_count
        second_correction = 1 - self.beta2 ** self.step_count
        nesterov = [
            (self.beta1 * first / first_correction)
            + ((1 - self.beta1) * gradient / first_correction)
            for first, gradient in zip(self.first_moment, updates)
        ]
        return [
            value * (1 - self.learning_rate * self.weight_decay)
            - self.learning_rate * direction / (math.sqrt(second / second_correction) + self.epsilon)
            for value, direction, second in zip(values, nesterov, self.second_moment)
        ]

    def reset(self) -> None:
        """Reset moments and step counter."""
        self.first_moment.clear()
        self.second_moment.clear()
        self.step_count = 0


@dataclass
class Lion(OptimizerStateMixin):
    """Lion optimizer using sign-based momentum updates and decoupled decay."""

    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.0
    momentum: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        _positive(self.learning_rate, "learning_rate")
        for name, value in (("beta1", self.beta1), ("beta2", self.beta2)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
                raise ValueError(f"{name} must be between zero and one")
        if isinstance(self.weight_decay, bool) or not isinstance(self.weight_decay, (int, float)):
            raise TypeError("weight_decay must be a number")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative and finite")

    def step(self, parameters: Sequence[float], gradients: Sequence[float]) -> list[float]:
        """Return parameters after one Lion sign-momentum update."""
        values = _vector(parameters, "parameters")
        updates = _vector(gradients, "gradients")
        if len(values) != len(updates):
            raise ValueError("parameters and gradients must have the same length")
        if self.momentum and len(self.momentum) != len(values):
            raise ValueError("parameter dimension changed after optimizer initialization")
        if not self.momentum:
            self.momentum = [0.0] * len(values)
        mixed = [self.beta1 * old + (1 - self.beta1) * gradient for old, gradient in zip(self.momentum, updates)]
        result = [
            value * (1 - self.learning_rate * self.weight_decay)
            - self.learning_rate * (1.0 if direction > 0 else -1.0 if direction < 0 else 0.0)
            for value, direction in zip(values, mixed)
        ]
        self.momentum = [self.beta2 * old + (1 - self.beta2) * gradient for old, gradient in zip(self.momentum, updates)]
        return result

    def reset(self) -> None:
        """Clear the running sign-momentum state."""
        self.momentum.clear()


@dataclass
class SAM(OptimizerStateMixin):
    """Sharpness-Aware Minimization optimizer with explicit two-step updates.

    Call :meth:`first_step` with the gradients at the current parameters, run a
    second forward/backward pass at the returned perturbed parameters, then call
    :meth:`second_step` with the second gradients. The perturbation is cleared
    after a successful second step.
    """

    learning_rate: float = 1e-2
    rho: float = 0.05
    adaptive: bool = False
    perturbation: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        _positive(self.learning_rate, "learning_rate")
        _positive(self.rho, "rho")
        if not isinstance(self.adaptive, bool):
            raise TypeError("adaptive must be a boolean")

    def _validate_pair(self, parameters: Sequence[float], gradients: Sequence[float]) -> tuple[list[float], list[float]]:
        values = _vector(parameters, "parameters")
        updates = _vector(gradients, "gradients")
        if len(values) != len(updates):
            raise ValueError("parameters and gradients must have the same length")
        return values, updates

    def first_step(self, parameters: Sequence[float], gradients: Sequence[float]) -> list[float]:
        """Return perturbed parameters for SAM's second forward/backward pass."""
        if self.perturbation:
            raise RuntimeError("first_step called before second_step")
        values, updates = self._validate_pair(parameters, gradients)
        scaled = [gradient * value if self.adaptive else gradient for value, gradient in zip(values, updates)]
        norm = math.sqrt(sum(value * value for value in scaled))
        scale = self.rho / norm if norm > 0 else 0.0
        self.perturbation = [scale * value for value in scaled]
        return [value + delta for value, delta in zip(values, self.perturbation)]

    def second_step(self, parameters: Sequence[float], gradients: Sequence[float]) -> list[float]:
        """Apply the second-pass gradient at the original parameter location."""
        values, updates = self._validate_pair(parameters, gradients)
        if not self.perturbation:
            raise RuntimeError("second_step requires a preceding first_step")
        if len(self.perturbation) != len(values):
            raise ValueError("parameter dimension changed between SAM steps")
        result = [
            value - delta - self.learning_rate * gradient
            for value, delta, gradient in zip(values, self.perturbation, updates)
        ]
        self.perturbation.clear()
        return result

    def reset(self) -> None:
        """Discard a pending SAM perturbation."""
        self.perturbation.clear()


@dataclass
class RAdam(OptimizerStateMixin):
    """Rectified Adam with variance-aware early-training updates."""

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
        if isinstance(self.weight_decay, bool) or not isinstance(self.weight_decay, (int, float)):
            raise TypeError("weight_decay must be a number")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative and finite")

    def step(self, parameters: Sequence[float], gradients: Sequence[float]) -> list[float]:
        """Return parameters after one variance-rectified Adam update."""
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
        rho_inf = 2 / (1 - self.beta2) - 1
        rho = rho_inf - 2 * self.step_count * self.beta2**self.step_count / correction2
        if rho > 4:
            rectification = math.sqrt((rho - 4) * (rho - 2) * rho_inf / ((rho_inf - 4) * rho * (rho_inf - 2)))
            step_size = self.learning_rate * rectification * math.sqrt(correction2) / correction1
            directions = [first / math.sqrt(second) for first, second in zip(self.first_moment, [value / correction2 for value in self.second_moment])]
        else:
            step_size = self.learning_rate / correction1
            directions = [first for first in self.first_moment]
        return [value * (1 - self.learning_rate * self.weight_decay) - step_size * direction for value, direction in zip(values, directions)]

    def reset(self) -> None:
        """Reset moments and step counter."""
        self.first_moment.clear()
        self.second_moment.clear()
        self.step_count = 0


@dataclass
class AdaGrad(OptimizerStateMixin):
    """AdaGrad with optional L2 weight decay for sparse or noisy gradients."""

    learning_rate: float = 1e-2
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    squared_gradient: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        _positive(self.learning_rate, "learning_rate")
        _positive(self.epsilon, "epsilon")
        if isinstance(self.weight_decay, bool) or not isinstance(self.weight_decay, (int, float)):
            raise TypeError("weight_decay must be a number")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative and finite")

    def step(self, parameters: Sequence[float], gradients: Sequence[float]) -> list[float]:
        """Return parameters after one accumulated-squared-gradient update."""
        values = _vector(parameters, "parameters")
        updates = _vector(gradients, "gradients")
        if len(values) != len(updates):
            raise ValueError("parameters and gradients must have the same length")
        if self.squared_gradient and len(self.squared_gradient) != len(values):
            raise ValueError("parameter dimension changed after optimizer initialization")
        if not self.squared_gradient:
            self.squared_gradient = [0.0] * len(values)
        result = []
        for index, (value, gradient) in enumerate(zip(values, updates)):
            adjusted = gradient + float(self.weight_decay) * value
            self.squared_gradient[index] += adjusted * adjusted
            denominator = math.sqrt(self.squared_gradient[index]) + self.epsilon
            result.append(value - self.learning_rate * adjusted / denominator)
        return result

    def reset(self) -> None:
        """Clear the accumulated squared-gradient state."""
        self.squared_gradient.clear()


@dataclass
class RMSprop(OptimizerStateMixin):
    """RMSprop with optional momentum and centered variance estimates."""

    learning_rate: float = 1e-2
    alpha: float = 0.99
    epsilon: float = 1e-8
    momentum: float = 0.0
    weight_decay: float = 0.0
    centered: bool = False
    square_average: list[float] = field(default_factory=list)
    momentum_buffer: list[float] = field(default_factory=list)
    gradient_average: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        _positive(self.learning_rate, "learning_rate")
        for name, value in (("alpha", self.alpha), ("momentum", self.momentum)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
                raise ValueError(f"{name} must be between zero and one")
        _positive(self.epsilon, "epsilon")
        if isinstance(self.weight_decay, bool) or not isinstance(self.weight_decay, (int, float)) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not isinstance(self.centered, bool):
            raise TypeError("centered must be a boolean")

    def step(self, parameters: Sequence[float], gradients: Sequence[float]) -> list[float]:
        """Return parameters after one RMSprop update."""
        values = _vector(parameters, "parameters")
        updates = _vector(gradients, "gradients")
        if len(values) != len(updates):
            raise ValueError("parameters and gradients must have the same length")
        if self.square_average and len(self.square_average) != len(values):
            raise ValueError("parameter dimension changed after optimizer initialization")
        if not self.square_average:
            self.square_average = [0.0] * len(values)
            self.momentum_buffer = [0.0] * len(values)
            self.gradient_average = [0.0] * len(values)
        result = []
        for index, (value, gradient) in enumerate(zip(values, updates)):
            adjusted = gradient + float(self.weight_decay) * value
            self.square_average[index] = self.alpha * self.square_average[index] + (1 - self.alpha) * adjusted**2
            if self.centered:
                self.gradient_average[index] = self.alpha * self.gradient_average[index] + (1 - self.alpha) * adjusted
                denominator = self.square_average[index] - self.gradient_average[index] ** 2 + self.epsilon
            else:
                denominator = self.square_average[index] + self.epsilon
            direction = adjusted / math.sqrt(max(denominator, 0.0))
            if self.momentum:
                self.momentum_buffer[index] = self.momentum * self.momentum_buffer[index] + direction
                direction = self.momentum_buffer[index]
            result.append(value - self.learning_rate * direction)
        return result

    def reset(self) -> None:
        """Clear running variance, mean, and momentum state."""
        self.square_average.clear()
        self.momentum_buffer.clear()
        self.gradient_average.clear()


__all__ = ["AdaGrad", "Adam", "Lion", "NAdam", "RAdam", "RMSprop", "SAM", "SGD"]
