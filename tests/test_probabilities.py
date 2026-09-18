import math

import pytest

from deeplearn_utils import (
    binary_cross_entropy,
    categorical_entropy,
    log_softmax,
    sigmoid,
    softmax,
    softmax_batch,
)


def test_sigmoid_is_stable_for_large_positive_and_negative_values():
    assert sigmoid(1000) == pytest.approx(1.0)
    assert sigmoid(-1000) == pytest.approx(0.0)
    assert sigmoid(0) == pytest.approx(0.5)


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


def test_binary_cross_entropy_clips_extreme_probabilities():
    assert binary_cross_entropy(1, 1.0) < 1e-10
    assert binary_cross_entropy(0, 0.0) < 1e-10
    assert binary_cross_entropy(1, 0.5) == pytest.approx(math.log(2))


@pytest.mark.parametrize("values", [[], "abc", [float("nan")], [True]])
def test_probability_transforms_validate_sequences(values):
    with pytest.raises((TypeError, ValueError)):
        softmax(values)


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
