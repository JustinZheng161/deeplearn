"""Dependency-free ranking metrics for binary classifiers."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _paired(scores: Sequence[float], labels: Sequence[int]) -> tuple[list[float], list[int]]:
    values, targets = list(scores), list(labels)
    if not values or len(values) != len(targets):
        raise ValueError("scores and labels must be non-empty and have the same length")
    for score in values:
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("scores must contain finite numbers")
    for label in targets:
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError("labels must contain only zero and one")
    if not any(targets) or all(targets):
        raise ValueError("labels must contain both positive and negative examples")
    return [float(score) for score in values], [int(label) for label in targets]


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Return the probability that a random positive ranks above a negative."""
    values, targets = _paired(scores, labels)
    order = sorted(range(len(values)), key=values.__getitem__)
    rank_sum = 0.0
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(targets[order[item]] for item in range(index, end))
        index = end
    positives = sum(targets)
    negatives = len(targets) - positives
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def precision_recall_curve(scores: Sequence[float], labels: Sequence[int]) -> tuple[list[float], list[float], list[float]]:
    """Return precision, recall, and descending score thresholds."""
    values, targets = _paired(scores, labels)
    order = sorted(range(len(values)), key=values.__getitem__, reverse=True)
    positives = sum(targets)
    true_positive = false_positive = 0
    precision, recall, thresholds = [], [], []
    for position, index in enumerate(order):
        true_positive += targets[index]
        false_positive += 1 - targets[index]
        if position == len(order) - 1 or values[index] != values[order[position + 1]]:
            precision.append(true_positive / (true_positive + false_positive))
            recall.append(true_positive / positives)
            thresholds.append(values[index])
    return precision, recall, thresholds


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Return the area under the precision-recall staircase."""
    precision, recall, _ = precision_recall_curve(scores, labels)
    previous_recall = 0.0
    area = 0.0
    for current_precision, current_recall in zip(precision, recall):
        area += current_precision * (current_recall - previous_recall)
        previous_recall = current_recall
    return area


def best_f1_threshold(scores: Sequence[float], labels: Sequence[int]) -> tuple[float, float]:
    """Return the score threshold and F1 maximizing binary predictions."""
    values, targets = _paired(scores, labels)
    thresholds = sorted(set(values), reverse=True)
    best_threshold, best_score = thresholds[-1], -1.0
    for threshold in thresholds:
        predicted = [score >= threshold for score in values]
        true_positive = sum(predicted[index] and targets[index] for index in range(len(values)))
        false_positive = sum(predicted[index] and not targets[index] for index in range(len(values)))
        false_negative = sum(not predicted[index] and targets[index] for index in range(len(values)))
        denominator = 2 * true_positive + false_positive + false_negative
        score = 2 * true_positive / denominator if denominator else 0.0
        if score > best_score:
            best_threshold, best_score = threshold, score
    return float(best_threshold), float(best_score)


__all__ = ["average_precision", "best_f1_threshold", "precision_recall_curve", "roc_auc"]
