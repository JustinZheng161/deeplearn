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


def one_cycle_lr(
    step: int,
    *,
    total_steps: int,
    max_lr: float,
    base_lr: float,
    final_lr: float,
    pct_start: float = 0.3,
) -> float:
    """Return a one-cycle learning rate with linear rise and cosine decay.

    The schedule starts at ``base_lr``, reaches ``max_lr`` at ``pct_start``
    of training, and finishes at ``final_lr``. Steps beyond the schedule are
    clamped to the final learning rate.
    """
    _validate_one_cycle(step, total_steps, max_lr, base_lr, final_lr, pct_start)
    if step >= total_steps - 1:
        return float(final_lr)
    peak_step = max(1, int(round((total_steps - 1) * float(pct_start))))
    if step <= peak_step:
        progress = step / peak_step
        return float(base_lr + (max_lr - base_lr) * progress)
    decay_progress = (step - peak_step) / (total_steps - 1 - peak_step)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return float(final_lr + (max_lr - final_lr) * cosine)


def _validate_one_cycle(
    step: int,
    total_steps: int,
    max_lr: float,
    base_lr: float,
    final_lr: float,
    pct_start: float,
) -> None:
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("step must be an integer")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int):
        raise TypeError("total_steps must be an integer")
    if step < 0:
        raise ValueError("step must be non-negative")
    if total_steps < 2:
        raise ValueError("total_steps must be at least two")
    if isinstance(pct_start, bool) or not isinstance(pct_start, (int, float)):
        raise TypeError("pct_start must be a number")
    if not math.isfinite(pct_start) or not 0 < pct_start < 1:
        raise ValueError("pct_start must be between zero and one")
    for name, value in (("max_lr", max_lr), ("base_lr", base_lr), ("final_lr", final_lr)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not base_lr <= max_lr:
        raise ValueError("base_lr must not exceed max_lr")
    if not final_lr <= max_lr:
        raise ValueError("final_lr must not exceed max_lr")


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
