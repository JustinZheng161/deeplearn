"""Learning-rate schedules that can be used with any optimizer framework."""

from __future__ import annotations

import math


def warmup_cosine_decay(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float = 0.0,
) -> float:
    """Return the learning rate for a linear-warmup cosine-decay schedule.

    ``step`` is zero-based and values beyond ``total_steps`` are clamped to
    ``min_lr``. Setting ``warmup_steps`` to zero starts immediately at
    ``peak_lr``. The function returns plain floats so callers can assign the
    result to PyTorch, TensorFlow, or custom optimizer state.
    """
    _validate_schedule(step, warmup_steps, total_steps, peak_lr, min_lr)
    if step >= total_steps:
        return float(min_lr)
    if warmup_steps and step < warmup_steps:
        return float(peak_lr * (step + 1) / warmup_steps)

    decay_steps = total_steps - warmup_steps
    decay_step = step - warmup_steps
    progress = decay_step / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr + (peak_lr - min_lr) * cosine)


def _validate_schedule(
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
) -> None:
    for name, value in (
        ("step", step),
        ("warmup_steps", warmup_steps),
        ("total_steps", total_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if step < 0:
        raise ValueError("step must be non-negative")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be less than total_steps")
    for name, value in (("peak_lr", peak_lr), ("min_lr", min_lr)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if peak_lr <= 0:
        raise ValueError("peak_lr must be positive")
    if min_lr < 0 or min_lr > peak_lr:
        raise ValueError("min_lr must be between zero and peak_lr")
