"""Dependency-light uncertainty measures for ensemble predictions."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


def _probability_rows(probabilities: Sequence[Sequence[float]]) -> list[list[float]]:
    if isinstance(probabilities, (str, bytes)):
        raise TypeError("probabilities must be a sequence of rows")
    rows = [list(row) for row in probabilities]
    if not rows:
        raise ValueError("probabilities must not be empty")
    width = len(rows[0])
    if width < 2 or any(len(row) != width for row in rows):
        raise ValueError("probabilities must be a rectangular matrix with at least two classes")
    for row in rows:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("probabilities must contain numbers")
            if not math.isfinite(value) or value < 0:
                raise ValueError("probabilities must be finite and non-negative")
        if not math.isclose(sum(row), 1.0, abs_tol=1e-8):
            raise ValueError("each probability row must sum to one")
    return [[float(value) for value in row] for row in rows]


def entropy(probabilities: Sequence[float], *, base: float = math.e) -> float:
    """Return Shannon entropy for one normalized probability vector."""
    values = list(probabilities)
    if not values or any(value < 0 or not math.isfinite(value) for value in values):
        raise ValueError("probabilities must be finite and non-negative")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-8):
        raise ValueError("probabilities must sum to one")
    if isinstance(base, bool) or not isinstance(base, (int, float)) or not math.isfinite(base):
        raise TypeError("base must be a finite number")
    if base <= 0 or base == 1:
        raise ValueError("base must be positive and different from one")
    return -sum(value * math.log(value, base) for value in values if value > 0)


def predictive_entropy(mean_probabilities: Sequence[float], *, base: float = math.e) -> float:
    """Return entropy of the ensemble-mean predictive distribution."""
    return entropy(mean_probabilities, base=base)


def ensemble_mean(probability_members: Sequence[Sequence[Sequence[float]]]) -> list[list[float]]:
    """Average member probabilities for each sample and class."""
    if isinstance(probability_members, (str, bytes)):
        raise TypeError("probability_members must be a sequence of members")
    members = [_probability_rows(member) for member in probability_members]
    if not members:
        raise ValueError("probability_members must not be empty")
    sample_count, class_count = len(members[0]), len(members[0][0])
    if any(len(member) != sample_count or len(member[0]) != class_count for member in members):
        raise ValueError("all ensemble members must have the same shape")
    return [[sum(member[sample][class_index] for member in members) / len(members) for class_index in range(class_count)] for sample in range(sample_count)]


def mutual_information(probability_members: Sequence[Sequence[Sequence[float]]], *, base: float = math.e) -> list[float]:
    """Estimate epistemic uncertainty as predictive minus expected entropy."""
    members = [_probability_rows(member) for member in probability_members]
    mean = ensemble_mean(members)
    expected = [sum(entropy(member[index], base=base) for member in members) / len(members) for index in range(len(mean))]
    return [predictive_entropy(row, base=base) - value for row, value in zip(mean, expected)]


def variation_ratio(predicted_classes: Sequence[int]) -> float:
    """Return one minus the fraction of ensemble votes for the modal class."""
    values = list(predicted_classes)
    if not values:
        raise ValueError("predicted_classes must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("predicted_classes must contain integer class indices")
    return 1.0 - max(Counter(values).values()) / len(values)


def risk_coverage_curve(
    confidences: Sequence[float], correct: Sequence[bool]
) -> list[tuple[float, float]]:
    """Return ``(coverage, risk)`` points for confidence-based abstention.

    Samples are retained from highest to lowest confidence. Each point reports
    the retained fraction and its error rate, making selective prediction
    trade-offs easy to plot or compare without a plotting dependency.
    """
    if isinstance(confidences, (str, bytes)) or isinstance(correct, (str, bytes)):
        raise TypeError("confidences and correct must be sequences")
    confidence_values = list(confidences)
    correctness = list(correct)
    if not confidence_values or len(confidence_values) != len(correctness):
        raise ValueError("confidences and correct must have the same non-zero length")
    for confidence in confidence_values:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError("confidences must contain numbers")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidences must be between zero and one")
    if any(not isinstance(value, bool) for value in correctness):
        raise TypeError("correct must contain booleans")
    order = sorted(range(len(confidence_values)), key=lambda index: confidence_values[index], reverse=True)
    errors = 0
    points = []
    for retained, index in enumerate(order, start=1):
        errors += not correctness[index]
        points.append((retained / len(order), errors / retained))
    return points


@dataclass(frozen=True)
class UncertaintySummary:
    """Per-sample ensemble uncertainty summary."""

    mean_probabilities: list[float]
    predictive_entropy: float
    mutual_information: float
    variation_ratio: float


def summarize_ensemble(
    probability_members: Sequence[Sequence[Sequence[float]]], *, base: float = math.e
) -> list[UncertaintySummary]:
    """Summarize predictive and epistemic uncertainty for each sample."""
    members = [_probability_rows(member) for member in probability_members]
    mean = ensemble_mean(members)
    information = mutual_information(members, base=base)
    summaries = []
    for sample_index, row in enumerate(mean):
        votes = [member[sample_index].index(max(member[sample_index])) for member in members]
        summaries.append(UncertaintySummary(row, entropy(row, base=base), information[sample_index], variation_ratio(votes)))
    return summaries


__all__ = [
    "UncertaintySummary",
    "ensemble_mean",
    "entropy",
    "mutual_information",
    "predictive_entropy",
    "risk_coverage_curve",
    "summarize_ensemble",
    "variation_ratio",
]
