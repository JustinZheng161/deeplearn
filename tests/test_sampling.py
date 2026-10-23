import pytest

from deeplearn_utils import categorical_sample, nucleus_filter, sample_logits, top_k_filter, top_p_sample


def test_categorical_sampling_is_reproducible_and_valid():
    first = categorical_sample([0.1, 0.2, 0.7], seed=4)
    assert first == categorical_sample([0.1, 0.2, 0.7], seed=4)
    assert first in {0, 1, 2}


def test_sample_logits_temperature_and_seed_are_supported():
    assert sample_logits([1, 2, 3], seed=10) == sample_logits([1, 2, 3], seed=10)
    assert sample_logits([1000, 0], seed=1) == 0


def test_top_k_filter_keeps_exactly_k_logits():
    filtered = top_k_filter([1, 5, 3, 2], 2)
    assert filtered[1] == 5.0
    assert filtered[2] == 3.0
    assert filtered[0] == float("-inf")
    assert filtered[3] == float("-inf")


def test_nucleus_filter_keeps_probability_mass_in_descending_order():
    filtered = nucleus_filter([5, 4, 1], probability=0.8)
    assert filtered[0] == 5.0
    assert filtered[1] == 4.0
    assert filtered[2] == float("-inf")


def test_top_p_sampling_returns_allowed_class_and_is_reproducible():
    first = top_p_sample([5, 4, 1], probability=0.8, seed=3)
    assert first == top_p_sample([5, 4, 1], probability=0.8, seed=3)
    assert first in {0, 1}


def test_sampling_validates_probability_vectors():
    with pytest.raises(ValueError):
        categorical_sample([0.2, 0.2], seed=1)
    with pytest.raises(ValueError):
        categorical_sample([1.1, -0.1])


@pytest.mark.parametrize("k", [0, 4, True, 1.5])
def test_top_k_validates_k(k):
    with pytest.raises(ValueError):
        top_k_filter([1, 2, 3], k)


@pytest.mark.parametrize("probability", [0, -0.1, 1.1, True])
def test_nucleus_validates_probability(probability):
    with pytest.raises(ValueError):
        nucleus_filter([1, 2], probability)


def test_sampling_validates_temperature_and_seed():
    with pytest.raises(ValueError):
        sample_logits([1, 2], temperature=0)
    with pytest.raises(TypeError):
        sample_logits([1, 2], seed=True)
