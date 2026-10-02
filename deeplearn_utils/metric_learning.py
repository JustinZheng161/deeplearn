"""Dependency-light metric-learning losses for embedding models."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _vectors(embeddings: Sequence[Sequence[float]]) -> list[list[float]]:
    if isinstance(embeddings, (str, bytes)):
        raise TypeError("embeddings must be a sequence of vectors")
    rows = [list(row) for row in embeddings]
    if not rows:
        raise ValueError("embeddings must not be empty")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("embeddings must be a non-empty rectangular matrix")
    for row in rows:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("embeddings must contain numbers")
            if not math.isfinite(value):
                raise ValueError("embeddings must contain finite values")
    return [[float(value) for value in row] for row in rows]


def _distance(left: Sequence[float], right: Sequence[float], squared: bool = False) -> float:
    value = sum((a - b) ** 2 for a, b in zip(left, right))
    return value if squared else math.sqrt(value)


def contrastive_loss(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    margin: float = 1.0,
    squared: bool = False,
) -> float:
    """Return pairwise contrastive loss for similar (1) and dissimilar (0) pairs."""
    first = _vectors(left)
    second = _vectors(right)
    targets = list(labels)
    if len(first) != len(second) or len(first) != len(targets):
        raise ValueError("left, right, and labels must have the same length")
    if len(first[0]) != len(second[0]):
        raise ValueError("embedding vectors must have the same dimension")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be positive and finite")
    losses = []
    for one, two, label in zip(first, second, targets):
        if isinstance(label, bool) or label not in {0, 1}:
            raise ValueError("labels must contain only zero or one")
        distance = _distance(one, two, squared=squared)
        negative = max(0.0, float(margin) - distance) ** 2
        losses.append(distance if label and squared else distance ** 2 if label else negative)
    return sum(losses) / len(losses)


def triplet_margin_loss(
    anchors: Sequence[Sequence[float]],
    positives: Sequence[Sequence[float]],
    negatives: Sequence[Sequence[float]],
    *,
    margin: float = 1.0,
    squared: bool = False,
) -> float:
    """Return mean hinge loss enforcing positives closer than negatives."""
    first = _vectors(anchors)
    positive = _vectors(positives)
    negative = _vectors(negatives)
    if not len(first) == len(positive) == len(negative):
        raise ValueError("anchors, positives, and negatives must have the same length")
    dimensions = len(first[0])
    if any(len(row) != dimensions for row in positive + negative):
        raise ValueError("embedding vectors must have the same dimension")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be positive and finite")
    losses = []
    for anchor, close, far in zip(first, positive, negative):
        positive_distance = _distance(anchor, close, squared=squared)
        negative_distance = _distance(anchor, far, squared=squared)
        losses.append(max(0.0, positive_distance - negative_distance + float(margin)))
    return sum(losses) / len(losses)


def pairwise_distance_matrix(
    embeddings: Sequence[Sequence[float]], *, squared: bool = False
) -> list[list[float]]:
    """Return a symmetric pairwise Euclidean distance matrix."""
    rows = _vectors(embeddings)
    return [[_distance(left, right, squared=squared) for right in rows] for left in rows]


def batch_hard_triplet_loss(
    embeddings: Sequence[Sequence[float]], labels: Sequence[int], *, margin: float = 1.0, squared: bool = False
) -> float:
    """Use hardest positive and hardest negative for each valid batch anchor."""
    rows = _vectors(embeddings)
    targets = list(labels)
    if len(rows) != len(targets):
        raise ValueError("embeddings and labels must have the same length")
    if any(isinstance(label, bool) or not isinstance(label, int) for label in targets):
        raise TypeError("labels must contain integer class indices")
    distances = pairwise_distance_matrix(rows, squared=squared)
    losses = []
    for index, label in enumerate(targets):
        positives = [distances[index][other] for other, value in enumerate(targets) if other != index and value == label]
        negatives = [distances[index][other] for other, value in enumerate(targets) if value != label]
        if positives and negatives:
            losses.append(max(0.0, max(positives) - min(negatives) + margin))
    if not losses:
        raise ValueError("batch must contain an anchor with both positive and negative examples")
    return sum(losses) / len(losses)


__all__ = ["batch_hard_triplet_loss", "contrastive_loss", "pairwise_distance_matrix", "triplet_margin_loss"]
