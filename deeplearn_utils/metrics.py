"""Dependency-light metrics for classification experiments.

The functions in this module accept ordinary Python sequences, making them
useful in evaluation code that should not require NumPy or a deep-learning
framework. Inputs are validated explicitly so that silent shape mistakes do
not corrupt experiment reports.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any


def _as_labels(values: Sequence[Any], name: str) -> list[Any]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of labels")
    try:
        labels = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of labels") from exc
    if not labels:
        raise ValueError(f"{name} must not be empty")
    return labels


def _validate_pair(y_true: Sequence[Any], y_pred: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    actual = _as_labels(y_true, "y_true")
    predicted = _as_labels(y_pred, "y_pred")
    if len(actual) != len(predicted):
        raise ValueError("y_true and y_pred must have the same length")
    return actual, predicted


def accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Return the fraction of predictions equal to their target labels."""
    actual, predicted = _validate_pair(y_true, y_pred)
    return sum(target == guess for target, guess in zip(actual, predicted)) / len(actual)


def confusion_matrix(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
) -> list[list[int]]:
    """Return a row-normalized-by-convention confusion count matrix.

    Rows represent actual labels and columns represent predicted labels. The
    label order is explicit when ``labels`` is supplied; otherwise labels are
    discovered in first-seen order across the target and prediction arrays.
    """
    actual, predicted = _validate_pair(y_true, y_pred)
    if labels is None:
        classes: list[Any] = []
        for value in [*actual, *predicted]:
            if value not in classes:
                classes.append(value)
    else:
        classes = _as_labels(labels, "labels")
        if len(set(classes)) != len(classes):
            raise ValueError("labels must not contain duplicates")
        unknown = [value for value in [*actual, *predicted] if value not in classes]
        if unknown:
            raise ValueError(f"labels do not include observed value: {unknown[0]!r}")
    index = {label: position for position, label in enumerate(classes)}
    matrix = [[0 for _ in classes] for _ in classes]
    for target, guess in zip(actual, predicted):
        matrix[index[target]][index[guess]] += 1
    return matrix


def precision_recall_f1(
    y_true: Sequence[Any], y_pred: Sequence[Any], *, positive_label: Any = 1
) -> dict[str, float]:
    """Return binary precision, recall, F1, and support for ``positive_label``.

    A zero denominator produces ``0.0`` rather than NaN, which keeps metrics
    serializable and makes all-negative validation batches safe to report.
    """
    actual, predicted = _validate_pair(y_true, y_pred)
    true_positive = sum(target == positive_label and guess == positive_label for target, guess in zip(actual, predicted))
    false_positive = sum(target != positive_label and guess == positive_label for target, guess in zip(actual, predicted))
    false_negative = sum(target == positive_label and guess != positive_label for target, guess in zip(actual, predicted))
    support = sum(target == positive_label for target in actual)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / support if support else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": float(support)}


def top_k_accuracy(
    y_true: Sequence[Any], scores: Sequence[Sequence[float]], *, k: int = 1
) -> float:
    """Return the fraction whose target is among the top ``k`` class scores.

    ``scores[i]`` must list one score per class in a stable class-index order;
    ties are resolved by lower class index, matching Python's stable sort.
    """
    actual = _as_labels(y_true, "y_true")
    if isinstance(scores, (str, bytes)):
        raise TypeError("scores must be a sequence of score rows")
    rows = list(scores)
    if len(rows) != len(actual):
        raise ValueError("y_true and scores must have the same length")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k < 1:
        raise ValueError("k must be positive")
    hits = 0
    class_count: int | None = None
    for target, row in zip(actual, rows):
        if isinstance(row, (str, bytes)):
            raise TypeError("each score row must be a sequence of numbers")
        values = list(row)
        if not values:
            raise ValueError("score rows must not be empty")
        if class_count is None:
            class_count = len(values)
        elif len(values) != class_count:
            raise ValueError("all score rows must have the same class count")
        if isinstance(target, bool) or not isinstance(target, int):
            raise TypeError("target labels must be integer class indices")
        if target < 0 or target >= len(values):
            raise ValueError("target label is outside the score class range")
        numeric = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError("scores must be finite numbers")
            numeric.append(float(value))
        top_indices = sorted(range(len(numeric)), key=lambda index: numeric[index], reverse=True)[:k]
        hits += target in top_indices
    return hits / len(actual)


def macro_f1(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Return the unweighted mean of per-class F1 scores."""
    actual, predicted = _validate_pair(y_true, y_pred)
    classes: list[Any] = []
    for value in [*actual, *predicted]:
        if value not in classes:
            classes.append(value)
    scores = [precision_recall_f1(actual, predicted, positive_label=label)["f1"] for label in classes]
    return sum(scores) / len(scores)


def balanced_accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Return the mean recall across observed classes.

    Each actual class receives equal weight, making this metric useful for
    imbalanced validation sets. A class missed entirely contributes zero.
    """
    actual, predicted = _validate_pair(y_true, y_pred)
    classes: list[Any] = []
    for value in actual:
        if value not in classes:
            classes.append(value)
    recalls = []
    for label in classes:
        support = sum(value == label for value in actual)
        correct = sum(
            target == label and guess == label
            for target, guess in zip(actual, predicted)
        )
        recalls.append(correct / support if support else 0.0)
    return sum(recalls) / len(recalls)


__all__ = [
    "accuracy",
    "balanced_accuracy",
    "confusion_matrix",
    "macro_f1",
    "precision_recall_f1",
    "top_k_accuracy",
]
