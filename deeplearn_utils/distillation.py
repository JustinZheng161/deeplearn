"""Dependency-free knowledge-distillation loss helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _rows(values: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of rows")
    rows = [list(row) for row in values]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{name} must be a non-empty rectangular matrix")
    for row in rows:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must contain finite numbers")
    return [[float(value) for value in row] for row in rows]


def _temperature(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("temperature must be positive and finite")


def _softmax(row: Sequence[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in row]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def soft_targets(logits: Sequence[Sequence[float]], *, temperature: float = 1.0) -> list[list[float]]:
    """Convert teacher logits to temperature-softened probability targets."""
    rows = _rows(logits, "logits")
    _temperature(temperature)
    return [_softmax(row, float(temperature)) for row in rows]


def distillation_kl(
    student_logits: Sequence[Sequence[float]], teacher_logits: Sequence[Sequence[float]], *, temperature: float = 1.0
) -> float:
    """Return batch-mean teacher-to-student KL loss scaled by T squared."""
    student = _rows(student_logits, "student_logits")
    teacher = _rows(teacher_logits, "teacher_logits")
    if len(student) != len(teacher) or len(student[0]) != len(teacher[0]):
        raise ValueError("student and teacher logits must have the same shape")
    _temperature(temperature)
    temperature = float(temperature)
    losses = []
    for student_row, teacher_row in zip(student, teacher):
        student_probabilities = _softmax(student_row, temperature)
        teacher_probabilities = _softmax(teacher_row, temperature)
        losses.append(sum(target * math.log(target / max(prediction, 1e-15)) for target, prediction in zip(teacher_probabilities, student_probabilities) if target > 0))
    return sum(losses) / len(losses) * temperature**2


def hard_cross_entropy(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Return standard mean cross entropy from logits and class indices."""
    rows = _rows(logits, "logits")
    targets = list(labels)
    if len(rows) != len(targets):
        raise ValueError("logits and labels must have the same length")
    losses = []
    for row, target in zip(rows, targets):
        if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < len(row):
            raise ValueError("labels must be valid class indices")
        probabilities = _softmax(row, 1.0)
        losses.append(-math.log(max(probabilities[target], 1e-15)))
    return sum(losses) / len(losses)


def distillation_loss(
    student_logits: Sequence[Sequence[float]],
    teacher_logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> float:
    """Blend soft teacher KL loss with hard-label cross entropy."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    soft = distillation_kl(student_logits, teacher_logits, temperature=temperature)
    hard = hard_cross_entropy(student_logits, labels)
    return float(alpha) * soft + (1.0 - float(alpha)) * hard


def teacher_student_agreement(
    student_logits: Sequence[Sequence[float]], teacher_logits: Sequence[Sequence[float]]
) -> float:
    """Return the fraction of examples sharing the teacher's top class."""
    student = _rows(student_logits, "student_logits")
    teacher = _rows(teacher_logits, "teacher_logits")
    if len(student) != len(teacher) or len(student[0]) != len(teacher[0]):
        raise ValueError("student and teacher logits must have the same shape")
    return sum(max(range(len(left)), key=left.__getitem__) == max(range(len(right)), key=right.__getitem__) for left, right in zip(student, teacher)) / len(student)


__all__ = ["distillation_kl", "distillation_loss", "hard_cross_entropy", "soft_targets", "teacher_student_agreement"]
