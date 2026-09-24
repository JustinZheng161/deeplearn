import math

import pytest

from deeplearn_utils import (
    cosine_similarity,
    mean_average_precision,
    mean_reciprocal_rank,
    ranked_indices,
    recall_at_k,
    similarity_matrix,
)


def test_cosine_similarity_handles_aligned_and_zero_vectors():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [0, 0]) == pytest.approx(1.0)
    assert cosine_similarity([0, 0], [1, 0]) == pytest.approx(0.0)


def test_similarity_matrix_is_pairwise_and_symmetric_for_same_input():
    vectors = [[1, 0], [0, 1]]
    result = similarity_matrix(vectors, vectors)
    assert result[0] == pytest.approx([1.0, 0.0])
    assert result[1] == pytest.approx([0.0, 1.0])


def test_ranked_indices_is_stable_on_ties():
    assert ranked_indices([0.2, 0.9, 0.9, 0.1]) == [1, 2, 0, 3]
    assert ranked_indices([0.2, 0.9, 0.9], descending=False) == [0, 1, 2]


def test_recall_at_k_counts_relevant_results_per_query():
    ranked = [[2, 0, 1], [1, 3, 2]]
    relevant = [[0, 1], [2, 3]]
    assert recall_at_k(ranked, relevant, k=2) == pytest.approx(0.5)
    assert recall_at_k(ranked, relevant, k=3) == pytest.approx(1.0)


def test_mean_reciprocal_rank_finds_first_relevant_result():
    ranked = [[4, 2, 1], [0, 3, 1]]
    relevant = [[1, 2], [3]]
    assert mean_reciprocal_rank(ranked, relevant) == pytest.approx((0.5 + 0.5) / 2)


def test_mean_average_precision_handles_multiple_relevant_items():
    ranked = [[1, 0, 2], [2, 3, 1]]
    relevant = [[0, 1], [2, 3]]
    assert mean_average_precision(ranked, relevant) == pytest.approx((1.0 + 1.0) / 2)


def test_retrieval_metrics_return_zero_without_hits():
    ranked = [[0, 1], [2, 3]]
    relevant = [[4], [5]]
    assert recall_at_k(ranked, relevant) == 0.0
    assert mean_reciprocal_rank(ranked, relevant) == 0.0
    assert mean_average_precision(ranked, relevant) == 0.0


@pytest.mark.parametrize("vectors", [[], [[1, 2], [3]], "abc"])
def test_vector_functions_validate_shapes(vectors):
    with pytest.raises((TypeError, ValueError)):
        similarity_matrix(vectors, [[1, 2]])


def test_cosine_similarity_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="same dimension"):
        cosine_similarity([1], [1, 2])


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_recall_validates_k(k):
    with pytest.raises((TypeError, ValueError)):
        recall_at_k([[0]], [[0]], k=k)


def test_ranking_metrics_validate_duplicates_and_lengths():
    with pytest.raises(ValueError, match="duplicates"):
        mean_reciprocal_rank([[0, 0]], [[0]])
    with pytest.raises(ValueError, match="same non-zero"):
        recall_at_k([[0]], [])


def test_precision_and_ndcg_reward_relevant_items_near_the_top():
    from deeplearn_utils import ndcg_at_k, precision_at_k

    ranked = [[2, 0, 1], [1, 3, 2]]
    relevant = [[0, 1], [2, 3]]
    assert precision_at_k(ranked, relevant, k=2) == pytest.approx(0.5)
    ideal = 1.0 + 1.0 / math.log2(3)
    expected = (1.0 / math.log2(3) + 1.0 / math.log2(4)) / ideal
    assert ndcg_at_k(ranked, relevant, k=3) == pytest.approx(expected)


def test_precision_and_ndcg_return_zero_without_relevant_hits():
    from deeplearn_utils import ndcg_at_k, precision_at_k

    assert precision_at_k([[0, 1]], [[3]], k=2) == 0.0
    assert ndcg_at_k([[0, 1]], [[]], k=2) == 0.0
