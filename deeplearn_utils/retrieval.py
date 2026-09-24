"""Dependency-light retrieval metrics for learned embeddings."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def _vectors(values: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of vectors")
    try:
        rows = [list(row) for row in values]
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of vectors") from exc
    if not rows:
        raise ValueError(f"{name} must not be empty")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be a non-empty rectangular matrix")
    for row in rows:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} values must be numbers")
            if not math.isfinite(value):
                raise ValueError(f"{name} values must be finite")
    return [[float(value) for value in row] for row in rows]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity, treating two zero vectors as identical."""
    first = _vectors([left], "left")[0]
    second = _vectors([right], "right")[0]
    if len(first) != len(second):
        raise ValueError("vectors must have the same dimension")
    dot = sum(a * b for a, b in zip(first, second))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 1.0 if first_norm == second_norm == 0 else 0.0
    return dot / (first_norm * second_norm)


def similarity_matrix(
    queries: Sequence[Sequence[float]], candidates: Sequence[Sequence[float]]
) -> list[list[float]]:
    """Return cosine similarities for every query-candidate pair."""
    query_vectors = _vectors(queries, "queries")
    candidate_vectors = _vectors(candidates, "candidates")
    if len(query_vectors[0]) != len(candidate_vectors[0]):
        raise ValueError("queries and candidates must have the same dimension")
    return [[cosine_similarity(query, candidate) for candidate in candidate_vectors] for query in query_vectors]


def ranked_indices(scores: Sequence[float], *, descending: bool = True) -> list[int]:
    """Return score indices in stable ranked order."""
    values = list(scores)
    if not values:
        raise ValueError("scores must not be empty")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("scores must contain finite numbers")
    return sorted(range(len(values)), key=lambda index: values[index], reverse=descending)


def recall_at_k(
    ranked: Sequence[Sequence[int]], relevant: Sequence[Sequence[int]], *, k: int = 10
) -> float:
    """Return mean query recall among the first ``k`` ranked results."""
    rankings, relevant_sets = _validate_rankings(ranked, relevant, k)
    recalls = []
    for ranking, expected in zip(rankings, relevant_sets):
        recalls.append(len(set(ranking[:k]) & expected) / len(expected) if expected else 0.0)
    return sum(recalls) / len(recalls)


def precision_at_k(
    ranked: Sequence[Sequence[int]], relevant: Sequence[Sequence[int]], *, k: int = 10
) -> float:
    """Return mean precision among the first ``k`` ranked results."""
    rankings, relevant_sets = _validate_rankings(ranked, relevant, k)
    precisions = [
        len(set(ranking[:k]) & expected) / min(k, len(ranking))
        if ranking[:k] else 0.0
        for ranking, expected in zip(rankings, relevant_sets)
    ]
    return sum(precisions) / len(precisions)


def ndcg_at_k(
    ranked: Sequence[Sequence[int]], relevant: Sequence[Sequence[int]], *, k: int = 10
) -> float:
    """Return mean normalized discounted cumulative gain at ``k``.

    Relevance is binary: items in each corresponding ``relevant`` set receive
    one point. Queries without relevant items contribute zero rather than NaN.
    """
    rankings, relevant_sets = _validate_rankings(ranked, relevant, k)
    scores = []
    for ranking, expected in zip(rankings, relevant_sets):
        if not expected:
            scores.append(0.0)
            continue
        dcg = sum(
            1.0 / math.log2(position + 2)
            for position, item in enumerate(ranking[:k])
            if item in expected
        )
        ideal_hits = min(len(expected), k, len(ranking))
        ideal = sum(1.0 / math.log2(position + 2) for position in range(ideal_hits))
        scores.append(dcg / ideal if ideal else 0.0)
    return sum(scores) / len(scores)


def mean_reciprocal_rank(
    ranked: Sequence[Sequence[int]], relevant: Sequence[Sequence[int]]
) -> float:
    """Return the mean reciprocal rank of the first relevant result."""
    rankings, relevant_sets = _validate_rankings(ranked, relevant, 1)
    reciprocal = []
    for ranking, expected in zip(rankings, relevant_sets):
        reciprocal.append(next((1.0 / (position + 1) for position, item in enumerate(ranking) if item in expected), 0.0))
    return sum(reciprocal) / len(reciprocal)


def mean_average_precision(
    ranked: Sequence[Sequence[int]], relevant: Sequence[Sequence[int]]
) -> float:
    """Return mean average precision over ranked retrieval results."""
    rankings, relevant_sets = _validate_rankings(ranked, relevant, 1)
    average_precisions = []
    for ranking, expected in zip(rankings, relevant_sets):
        if not expected:
            average_precisions.append(0.0)
            continue
        hits = 0
        precision_sum = 0.0
        for position, item in enumerate(ranking, start=1):
            if item in expected:
                hits += 1
                precision_sum += hits / position
        average_precisions.append(precision_sum / len(expected))
    return sum(average_precisions) / len(average_precisions)


def _validate_rankings(
    ranked: Sequence[Sequence[int]], relevant: Sequence[Sequence[int]], k: int
) -> tuple[list[list[int]], list[set[int]]]:
    if isinstance(ranked, (str, bytes)) or isinstance(relevant, (str, bytes)):
        raise TypeError("ranked and relevant must be sequences")
    rankings = [list(row) for row in ranked]
    expected_rows = [set(row) for row in relevant]
    if not rankings or len(rankings) != len(expected_rows):
        raise ValueError("ranked and relevant must have the same non-zero length")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k < 1:
        raise ValueError("k must be positive")
    for row in rankings:
        if any(isinstance(item, bool) or not isinstance(item, int) for item in row):
            raise TypeError("ranked items must be integers")
        if len(set(row)) != len(row):
            raise ValueError("ranked rows must not contain duplicates")
    return rankings, expected_rows


__all__ = [
    "cosine_similarity",
    "mean_average_precision",
    "ndcg_at_k",
    "precision_at_k",
    "mean_reciprocal_rank",
    "ranked_indices",
    "recall_at_k",
    "similarity_matrix",
]
