"""Post-hoc calibration helpers for multiclass model logits."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .probabilities import log_softmax, softmax


def _logit_rows(logits: Sequence[Sequence[float]]) -> list[list[float]]:
    if isinstance(logits, (str, bytes)):
        raise TypeError("logits must be a sequence of rows")
    rows = [list(row) for row in logits]
    if not rows:
        raise ValueError("logits must not be empty")
    width = len(rows[0])
    if width < 2 or any(len(row) != width for row in rows):
        raise ValueError("logits must be a rectangular matrix with at least two classes")
    for row in rows:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("logits must contain numbers")
            if not math.isfinite(value):
                raise ValueError("logits must contain finite values")
    return [[float(value) for value in row] for row in rows]


def _labels(labels: Sequence[int], size: int, classes: int) -> list[int]:
    values = list(labels)
    if len(values) != size:
        raise ValueError("logits and labels must have the same length")
    for label in values:
        if isinstance(label, bool) or not isinstance(label, int):
            raise TypeError("labels must contain integer class indices")
        if label < 0 or label >= classes:
            raise ValueError("label is outside the logits class range")
    return values


def negative_log_likelihood(logits: Sequence[Sequence[float]], labels: Sequence[int], *, temperature: float = 1.0) -> float:
    """Return mean multiclass NLL after dividing logits by ``temperature``."""
    rows = _logit_rows(logits)
    targets = _labels(labels, len(rows), len(rows[0]))
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a number")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    losses = [-log_softmax([value / temperature for value in row])[target] for row, target in zip(rows, targets)]
    return sum(losses) / len(losses)


def calibrated_probabilities(logits: Sequence[Sequence[float]], temperature: float = 1.0) -> list[list[float]]:
    """Convert logits to probabilities after applying temperature scaling."""
    rows = _logit_rows(logits)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a number")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    return [softmax([value / temperature for value in row]) for row in rows]


@dataclass(frozen=True)
class CalibrationBin:
    """Summary of confidence and accuracy for one reliability bin."""

    lower: float
    upper: float
    count: int
    accuracy: float
    confidence: float


def reliability_bins(
    probabilities: Sequence[Sequence[float]], labels: Sequence[int], *, bins: int = 10
) -> list[CalibrationBin]:
    """Build equal-width reliability bins from multiclass probabilities."""
    rows = _logit_rows(probabilities)
    targets = _labels(labels, len(rows), len(rows[0]))
    if isinstance(bins, bool) or not isinstance(bins, int):
        raise TypeError("bins must be an integer")
    if bins < 1:
        raise ValueError("bins must be positive")
    buckets: list[list[tuple[bool, float]]] = [[] for _ in range(bins)]
    for row, target in zip(rows, targets):
        total = sum(math.exp(value - max(row)) for value in row)
        probabilities_row = [math.exp(value - max(row)) / total for value in row]
        confidence = max(probabilities_row)
        prediction = probabilities_row.index(confidence)
        index = min(int(confidence * bins), bins - 1)
        buckets[index].append((prediction == target, confidence))
    result = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        result.append(CalibrationBin(
            index / bins,
            (index + 1) / bins,
            len(bucket),
            sum(correct for correct, _ in bucket) / len(bucket),
            sum(confidence for _, confidence in bucket) / len(bucket),
        ))
    return result


class TemperatureScaler:
    """Fit one positive temperature with deterministic coordinate search."""

    def __init__(self, temperature: float = 1.0) -> None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be a number")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        self.temperature = float(temperature)
        self.fitted = False

    def fit(self, logits: Sequence[Sequence[float]], labels: Sequence[int], *, steps: int = 24) -> "TemperatureScaler":
        """Choose temperature minimizing validation negative log likelihood."""
        rows = _logit_rows(logits)
        targets = _labels(labels, len(rows), len(rows[0]))
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be an integer")
        if steps < 1:
            raise ValueError("steps must be positive")
        lower, upper = 0.05, 10.0
        best_temperature = self.temperature
        best_loss = negative_log_likelihood(rows, targets, temperature=best_temperature)
        for _ in range(steps):
            candidates = [lower, upper, math.sqrt(lower * upper)]
            candidates.extend(lower + (upper - lower) * index / 10 for index in range(1, 10))
            candidate, loss = min(
                ((value, negative_log_likelihood(rows, targets, temperature=value)) for value in candidates),
                key=lambda item: item[1],
            )
            if loss < best_loss:
                best_temperature, best_loss = candidate, loss
            if candidate <= math.sqrt(lower * upper):
                upper = math.sqrt(lower * upper)
            else:
                lower = math.sqrt(lower * upper)
        self.temperature = best_temperature
        self.fitted = True
        return self

    def transform(self, logits: Sequence[Sequence[float]]) -> list[list[float]]:
        """Return calibrated probabilities using the fitted temperature."""
        return calibrated_probabilities(logits, self.temperature)

    def fit_transform(self, logits: Sequence[Sequence[float]], labels: Sequence[int], *, steps: int = 24) -> list[list[float]]:
        """Fit on validation data and return its calibrated probabilities."""
        return self.fit(logits, labels, steps=steps).transform(logits)


__all__ = ["CalibrationBin", "TemperatureScaler", "calibrated_probabilities", "negative_log_likelihood", "reliability_bins"]
