import math

import pytest

from deeplearn_utils import (
    binary_cross_entropy,
    categorical_entropy,
    categorical_cross_entropy_with_logits,
    categorical_cross_entropy_with_logits_batch,
    label_smoothed_cross_entropy_with_logits,
    label_smoothed_cross_entropy_with_logits_batch,
    kl_divergence,
    log_softmax,
    sigmoid,
    softmax,
    sparsemax,
    softmax_batch,
    sparse_cross_entropy_with_logits,
    sparse_cross_entropy_with_logits_batch,
)


def test_sigmoid_is_stable_for_large_positive_and_negative_values():
    assert sigmoid(1000) == pytest.approx(1.0)
    assert sigmoid(-1000) == pytest.approx(0.0)
    assert sigmoid(0) == pytest.approx(0.5)


def test_sparsemax_is_normalized_and_produces_exact_zeros():
    values = sparsemax([3.0, 1.0, -2.0])
    assert values == pytest.approx([1.0, 0.0, 0.0])
    assert sum(values) == pytest.approx(1.0)
    assert sparsemax([101.0, 99.0, 96.0]) == pytest.approx(values)


def test_softmax_sums_to_one_and_is_shift_invariant():
    values = softmax([1.0, 2.0, 3.0])
    shifted = softmax([101.0, 102.0, 103.0])
    assert sum(values) == pytest.approx(1.0)
    assert values == pytest.approx(shifted)
    assert all(value > 0 for value in values)


def test_log_softmax_exponentiates_to_softmax():
    logits = [0.5, -1.0, 4.0]
    assert [math.exp(value) for value in log_softmax(logits)] == pytest.approx(softmax(logits))


def test_softmax_batch_processes_rows_independently():
    result = softmax_batch([[1, 2], [2, 1]])
    assert len(result) == 2
    assert result[0] == pytest.approx([result[1][1], result[1][0]])


def test_categorical_entropy_supports_nats_and_bits():
    assert categorical_entropy([0.5, 0.5]) == pytest.approx(math.log(2))
    assert categorical_entropy([0.5, 0.5], base=2) == pytest.approx(1.0)
    assert categorical_entropy([1.0, 0.0]) == pytest.approx(0.0)


def test_categorical_cross_entropy_with_logits_matches_probability_form():
    logits = [1.0, 2.0, 3.0]
    expected = -math.log(softmax(logits)[2])
    assert categorical_cross_entropy_with_logits([0.0, 0.0, 1.0], logits) == pytest.approx(expected)
    assert categorical_cross_entropy_with_logits([0.25, 0.75], [0.0, 0.0]) == pytest.approx(math.log(2))


def test_label_smoothed_logits_loss_matches_sparse_loss_without_smoothing():
    logits = [1.0, 2.0, 3.0]
    assert label_smoothed_cross_entropy_with_logits(logits, 2, smoothing=0.0) == pytest.approx(
        sparse_cross_entropy_with_logits(logits, 2)
    )


def test_label_smoothed_logits_loss_distributes_mass_over_other_classes():
    logits = [2.0, 0.0, -1.0]
    smoothing = 0.2
    expected = -(0.8 * log_softmax(logits)[0] + 0.1 * log_softmax(logits)[1] + 0.1 * log_softmax(logits)[2])
    assert label_smoothed_cross_entropy_with_logits(logits, 0, smoothing=smoothing) == pytest.approx(expected)
    assert label_smoothed_cross_entropy_with_logits_batch([logits, logits], [0, 0], smoothing=smoothing) == pytest.approx(expected)


def test_label_smoothed_logits_loss_is_stable_for_extreme_logits():
    assert math.isfinite(label_smoothed_cross_entropy_with_logits([1000.0, 0.0], 0))
    assert label_smoothed_cross_entropy_with_logits([1000.0, 0.0], 0, smoothing=0.0) == pytest.approx(0.0)


def test_label_smoothed_logits_loss_validates_configuration():
    with pytest.raises(ValueError, match="valid class index"):
        label_smoothed_cross_entropy_with_logits([1.0, 0.0], 2)
    with pytest.raises(ValueError, match="between zero and one"):
        label_smoothed_cross_entropy_with_logits([1.0, 0.0], 0, smoothing=1.0)
    with pytest.raises(TypeError, match="smoothing"):
        label_smoothed_cross_entropy_with_logits([1.0, 0.0], 0, smoothing=True)
    with pytest.raises(ValueError, match="batch size"):
        label_smoothed_cross_entropy_with_logits_batch([[1.0, 0.0]], [0, 1])


def test_categorical_cross_entropy_with_logits_is_stable_for_large_logits():
    assert categorical_cross_entropy_with_logits([1.0, 0.0], [1000.0, 0.0]) == pytest.approx(0.0)


def test_categorical_cross_entropy_with_logits_validates_targets():
    with pytest.raises(ValueError, match="same length"):
        categorical_cross_entropy_with_logits([1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="sum to one"):
        categorical_cross_entropy_with_logits([0.5, 0.4], [0.0, 1.0])


def test_kl_divergence_is_zero_for_identical_distributions():
    assert kl_divergence([0.25, 0.75], [0.25, 0.75]) == pytest.approx(0.0)


def test_kl_divergence_supports_soft_targets_and_zero_target_entries():
    expected = 0.25 * math.log(0.25 / 0.5) + 0.75 * math.log(0.75 / 0.5)
    assert kl_divergence([0.25, 0.75], [0.5, 0.5]) == pytest.approx(expected)
    assert kl_divergence([1.0, 0.0], [0.5, 0.5]) == pytest.approx(math.log(2))


def test_kl_divergence_validates_distributions_and_epsilon():
    with pytest.raises(ValueError):
        kl_divergence([0.5, 0.5], [1.0])
    with pytest.raises(ValueError):
        kl_divergence([0.5, 0.4], [0.5, 0.5])
    with pytest.raises(ValueError):
        kl_divergence([1.0, 0.0], [1.0, 0.0], epsilon=0)


def test_binary_cross_entropy_clips_extreme_probabilities():
    assert binary_cross_entropy(1, 1.0) < 1e-10
    assert binary_cross_entropy(0, 0.0) < 1e-10
    assert binary_cross_entropy(1, 0.5) == pytest.approx(math.log(2))


@pytest.mark.parametrize("values", [[], "abc", [float("nan")], [True]])
def test_probability_transforms_validate_sequences(values):
    with pytest.raises((TypeError, ValueError)):
        softmax(values)
    with pytest.raises((TypeError, ValueError)):
        sparsemax(values)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "1"])
def test_sigmoid_validates_input(value):
    with pytest.raises((TypeError, ValueError)):
        sigmoid(value)


@pytest.mark.parametrize("probabilities", [[0.2, 0.2], [-0.1, 1.1], [True, 0.0]])
def test_entropy_validates_probability_distribution(probabilities):
    with pytest.raises((TypeError, ValueError)):
        categorical_entropy(probabilities)


def test_entropy_validates_base():
    with pytest.raises(ValueError):
        categorical_entropy([1.0], base=1)
    with pytest.raises(TypeError):
        categorical_entropy([1.0], base=True)


@pytest.mark.parametrize(
    ("target", "probability", "error"),
    [(-1, 0.5, ValueError), (1, 2, ValueError), (True, 0.5, TypeError)],
)
def test_binary_cross_entropy_validates_inputs(target, probability, error):
    with pytest.raises(error):
        binary_cross_entropy(target, probability)


def test_logits_cross_entropy_batch_averages_soft_targets():
    targets = [[1.0, 0.0], [0.0, 1.0]]
    logits = [[4.0, 0.0], [0.0, 4.0]]
    expected = categorical_cross_entropy_with_logits(targets[0], logits[0])
    assert categorical_cross_entropy_with_logits_batch(targets, logits) == pytest.approx(expected)


def test_sparse_logits_cross_entropy_matches_probability_form():
    logits = [1.0, 2.0, 3.0]
    assert sparse_cross_entropy_with_logits(logits, 2) == pytest.approx(
        -math.log(softmax(logits)[2])
    )
    assert sparse_cross_entropy_with_logits_batch([logits, logits], [2, 2]) == pytest.approx(
        sparse_cross_entropy_with_logits(logits, 2)
    )


def test_sparse_logits_loss_remains_stable_for_extreme_values():
    assert sparse_cross_entropy_with_logits([1000.0, 0.0], 0) == pytest.approx(0.0)
    assert sparse_cross_entropy_with_logits([-1000.0, 0.0], 0) == pytest.approx(1000.0)


def test_batch_logits_losses_validate_alignment_and_targets():
    with pytest.raises(ValueError, match="batch size"):
        categorical_cross_entropy_with_logits_batch([[1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="valid class index"):
        sparse_cross_entropy_with_logits([0.0, 1.0], 2)
    with pytest.raises(ValueError, match="batch size"):
        sparse_cross_entropy_with_logits_batch([[0.0, 1.0]], [0, 1])
