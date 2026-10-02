import pytest

from deeplearn_utils import (
    batch_hard_triplet_loss,
    contrastive_loss,
    pairwise_distance_matrix,
    triplet_margin_loss,
)


def test_pairwise_distance_matrix_is_symmetric():
    matrix = pairwise_distance_matrix([[0, 0], [3, 4], [0, 4]])
    assert matrix[0] == pytest.approx([0, 5, 4])
    assert matrix[1] == pytest.approx([5, 0, 3])
    assert matrix[2] == pytest.approx([4, 3, 0])


def test_contrastive_loss_rewards_similar_and_separated_pairs():
    similar = contrastive_loss([[0, 0]], [[0.1, 0]], [1])
    separated = contrastive_loss([[0, 0]], [[3, 0]], [0])
    violating = contrastive_loss([[0, 0]], [[0.2, 0]], [0])
    assert similar < 0.02
    assert separated == pytest.approx(0.0)
    assert violating > 0


def test_triplet_margin_loss_is_zero_when_ordering_has_margin():
    good = triplet_margin_loss([[0, 0]], [[0.1, 0]], [[3, 0]], margin=1)
    bad = triplet_margin_loss([[0, 0]], [[3, 0]], [[0.1, 0]], margin=1)
    assert good == pytest.approx(0.0)
    assert bad > 0


def test_squared_distance_options_are_supported():
    assert pairwise_distance_matrix([[0, 0], [3, 4]], squared=True) == [[0.0, 25.0], [25.0, 0.0]]
    assert contrastive_loss([[0, 0]], [[0.5, 0]], [1], squared=True) == pytest.approx(0.25)


def test_batch_hard_triplet_loss_uses_valid_anchors():
    embeddings = [[0, 0], [0.1, 0], [3, 0], [3.1, 0]]
    labels = [0, 0, 1, 1]
    assert batch_hard_triplet_loss(embeddings, labels, margin=1) == pytest.approx(0.0)


def test_batch_hard_triplet_rejects_batches_without_positive_or_negative():
    with pytest.raises(ValueError, match="positive"):
        batch_hard_triplet_loss([[0, 0], [1, 0]], [0, 1])
    with pytest.raises(ValueError, match="positive"):
        batch_hard_triplet_loss([[0, 0], [0.1, 0]], [0, 0])


def test_metric_losses_validate_lengths_and_labels():
    with pytest.raises(ValueError, match="same length"):
        contrastive_loss([[0]], [[0]], [])
    with pytest.raises(ValueError):
        contrastive_loss([[0]], [[0]], [2])
    with pytest.raises(ValueError):
        triplet_margin_loss([[0]], [], [[1]])


@pytest.mark.parametrize("embeddings", [[], [[1, 2], [3]], "abc"])
def test_distance_matrix_validates_embeddings(embeddings):
    with pytest.raises((TypeError, ValueError)):
        pairwise_distance_matrix(embeddings)


def test_metric_losses_validate_margin_and_dimensions():
    with pytest.raises(ValueError):
        contrastive_loss([[0]], [[1]], [1], margin=0)
    with pytest.raises(ValueError, match="dimension"):
        triplet_margin_loss([[0]], [[0, 1]], [[1]])
