import pytest

from deeplearn_utils import (
    add_gaussian_noise,
    compose_augmentations,
    mixup,
    random_feature_dropout,
)


FEATURES = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def test_zero_noise_returns_copy_with_same_values():
    source = [row[:] for row in FEATURES]
    result = add_gaussian_noise(source, stddev=0.0, seed=7)
    assert result == source
    assert result is not source
    assert result[0] is not source[0]


def test_noise_is_reproducible_and_does_not_change_input():
    source = [row[:] for row in FEATURES]
    first = add_gaussian_noise(source, stddev=0.5, seed=11)
    second = add_gaussian_noise(source, stddev=0.5, seed=11)
    assert first == second
    assert source == FEATURES
    assert first != FEATURES


def test_feature_dropout_probability_extremes():
    assert random_feature_dropout(FEATURES, probability=0.0, seed=3) == FEATURES
    assert random_feature_dropout(FEATURES, probability=1.0, seed=3) == [[0.0, 0.0]] * 3


def test_feature_dropout_is_reproducible():
    first = random_feature_dropout(FEATURES, probability=0.5, seed=4)
    assert first == random_feature_dropout(FEATURES, probability=0.5, seed=4)
    assert first != random_feature_dropout(FEATURES, probability=0.5, seed=5)


def test_mixup_preserves_batch_shape_and_target_weights():
    mixed, targets = mixup(FEATURES, [0, 1, 2], alpha=0.5, seed=8)
    assert len(mixed) == len(FEATURES)
    assert all(len(row) == 2 for row in mixed)
    assert len(targets) == 3
    for left, right, weight in targets:
        assert left in {0, 1, 2}
        assert right in {0, 1, 2}
        assert 0.0 <= weight <= 1.0


def test_mixup_is_reproducible_and_supports_self_pairs():
    first = mixup(FEATURES, ["a", "b", "c"], alpha=1.0, seed=2)
    assert first == mixup(FEATURES, ["a", "b", "c"], alpha=1.0, seed=2)


def test_composed_augmentations_are_deterministic():
    first = compose_augmentations(FEATURES, noise_stddev=0.1, dropout_probability=0.2, seed=4)
    assert first == compose_augmentations(FEATURES, noise_stddev=0.1, dropout_probability=0.2, seed=4)


@pytest.mark.parametrize("features", [[], [[1, 2], [3]], "abc"])
def test_augmentations_reject_empty_ragged_or_string_features(features):
    with pytest.raises((TypeError, ValueError)):
        add_gaussian_noise(features)


@pytest.mark.parametrize("probability", [-0.1, 1.1, True, "0.2"])
def test_dropout_validates_probability(probability):
    with pytest.raises((TypeError, ValueError)):
        random_feature_dropout(FEATURES, probability=probability)


@pytest.mark.parametrize("stddev", [-1, float("inf"), True, "0.1"])
def test_noise_validates_standard_deviation(stddev):
    with pytest.raises((TypeError, ValueError)):
        add_gaussian_noise(FEATURES, stddev=stddev)


def test_mixup_validates_targets_and_alpha():
    with pytest.raises(ValueError, match="same length"):
        mixup(FEATURES, [1])
    with pytest.raises(ValueError):
        mixup(FEATURES, [1, 2, 3], alpha=0)
    with pytest.raises(TypeError):
        mixup(FEATURES, [1, 2, 3], alpha=True)


def test_augmentations_reject_nonfinite_values():
    with pytest.raises(ValueError):
        add_gaussian_noise([[float("nan")]])
    with pytest.raises(TypeError):
        random_feature_dropout([["x"]])
