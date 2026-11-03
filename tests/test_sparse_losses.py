import math

import pytest

from deeplearn_utils import sparse_brier_score, sparse_cross_entropy, sparse_entropy, sparse_focal_loss, support_size, top_k_mass, top_probability


def test_sparse_cross_entropy_and_entropy_handle_zero_mass():
    assert sparse_cross_entropy([0.0, 1.0], 1) == pytest.approx(0.0)
    assert sparse_entropy([0.5, 0.5]) == pytest.approx(math.log(2))
    assert sparse_entropy([1.0, 0.0]) == pytest.approx(0.0)


def test_support_size_and_top_probability_describe_sparse_distribution():
    values = [0.0, 0.7, 0.0, 0.3]
    assert support_size(values) == 2
    assert support_size(values, tolerance=0.5) == 1
    assert top_probability(values) == (1, 0.7)


def test_top_k_mass_is_monotonic():
    values = [0.1, 0.6, 0.2, 0.1]
    assert top_k_mass(values, 1) == pytest.approx(0.6)
    assert top_k_mass(values, 2) == pytest.approx(0.8)
    assert top_k_mass(values, 4) == pytest.approx(1.0)


def test_sparse_focal_loss_downweights_easy_examples_and_supports_alpha():
    easy = sparse_focal_loss([0.9, 0.1], 0, gamma=2.0)
    hard = sparse_focal_loss([0.6, 0.4], 0, gamma=2.0)
    assert easy < hard
    assert sparse_focal_loss([0.5, 0.5], 0, gamma=0.0) == pytest.approx(math.log(2))
    assert sparse_focal_loss([0.5, 0.5], 0, gamma=0.0, alpha=0.25) == pytest.approx(math.log(2) / 4)


def test_sparse_brier_score_rewards_confident_correct_predictions():
    good = sparse_brier_score([0.9, 0.1], 0)
    bad = sparse_brier_score([0.1, 0.9], 0)
    assert good < bad
    assert good == pytest.approx(0.02)


def test_sparse_helpers_validate_normalization_and_targets():
    for function in (sparse_entropy, top_probability):
        with pytest.raises(ValueError):
            function([0.2, 0.2])
    with pytest.raises(ValueError):
        sparse_cross_entropy([0.5, 0.5], 2)
    with pytest.raises(ValueError):
        sparse_brier_score([0.5, 0.5], -1)


def test_sparse_helpers_validate_parameters():
    with pytest.raises(ValueError):
        support_size([1.0], tolerance=-1)
    with pytest.raises(ValueError):
        top_k_mass([1.0], 0)
    with pytest.raises(ValueError):
        sparse_entropy([float("nan"), 0.0])
    with pytest.raises(ValueError):
        sparse_focal_loss([0.5, 0.5], 0, gamma=-1)
    with pytest.raises(ValueError):
        sparse_focal_loss([0.5, 0.5], 0, alpha=-1)
