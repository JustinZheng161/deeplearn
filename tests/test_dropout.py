import pytest

from deeplearn_utils import dropout, dropout_mask, feature_dropout, stochastic_depth


def test_dropout_is_reproducible_and_uses_inverted_scaling():
    first = dropout([1, 2, 3, 4], rate=0.5, seed=7)
    second = dropout([1, 2, 3, 4], rate=0.5, seed=7)
    assert first == second
    assert all(value in {0.0, 2.0, 4.0, 6.0, 8.0} for value in first)


def test_dropout_is_identity_at_evaluation_and_zero_rate():
    values = [1, 2, 3]
    assert dropout(values, rate=0.5, seed=1, training=False) == [1.0, 2.0, 3.0]
    assert dropout(values, rate=0, seed=1) == [1.0, 2.0, 3.0]


def test_feature_dropout_shares_mask_across_rows():
    result = feature_dropout([[1, 2, 3], [4, 5, 6]], rate=0.5, seed=4)
    for column in range(3):
        assert (result[0][column] == 0) == (result[1][column] == 0)


def test_stochastic_depth_is_reproducible_and_preserves_shortcut():
    residual, shortcut = [2, 4], [1, 1]
    assert stochastic_depth(residual, shortcut, drop_rate=0.999, seed=1) == shortcut
    kept = stochastic_depth(residual, shortcut, drop_rate=0, seed=1)
    assert kept == [3.0, 5.0]
    assert stochastic_depth(residual, shortcut, drop_rate=0.5, seed=3) == stochastic_depth(residual, shortcut, drop_rate=0.5, seed=3)


def test_dropout_mask_is_reproducible():
    assert dropout_mask(5, rate=0.2, seed=9) == dropout_mask(5, rate=0.2, seed=9)
    assert len(dropout_mask(5, rate=0.2, seed=9)) == 5


@pytest.mark.parametrize("rate", [-0.1, 1, True, "0.5"])
def test_dropout_validates_rate(rate):
    with pytest.raises((TypeError, ValueError)):
        dropout([1], rate=rate)


def test_dropout_validates_seeds_and_training_flag():
    with pytest.raises(TypeError):
        dropout([1], seed=True)
    with pytest.raises(TypeError):
        dropout([1], training=1)


def test_feature_dropout_validates_matrix_and_stochastic_dimensions():
    with pytest.raises(ValueError):
        feature_dropout([[1], [2, 3]])
    with pytest.raises(ValueError):
        stochastic_depth([1], [1, 2])
    with pytest.raises(ValueError):
        dropout_mask(0)
