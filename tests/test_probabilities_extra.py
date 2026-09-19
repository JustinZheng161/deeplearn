import math

import pytest

from deeplearn_utils import categorical_cross_entropy, categorical_cross_entropy_batch


def test_categorical_cross_entropy_supports_hard_and_soft_targets():
    assert categorical_cross_entropy([1, 0, 0], [0.8, 0.1, 0.1]) == pytest.approx(-math.log(0.8))
    assert categorical_cross_entropy([0.8, 0.2], [0.6, 0.4]) == pytest.approx(
        -(0.8 * math.log(0.6) + 0.2 * math.log(0.4))
    )


def test_batch_cross_entropy_returns_mean_and_clips_zero_probability():
    value = categorical_cross_entropy_batch([[1, 0], [0, 1]], [[1, 0], [0, 1]])

    assert value == pytest.approx(-math.log(1.0))
    assert math.isfinite(categorical_cross_entropy([1, 0], [0, 1]))


def test_categorical_cross_entropy_validates_distributions():
    with pytest.raises(ValueError, match="same length"):
        categorical_cross_entropy([1, 0], [1])
    with pytest.raises(ValueError, match="sum to one"):
        categorical_cross_entropy([1, 1], [0.5, 0.5])
    with pytest.raises(ValueError, match="between"):
        categorical_cross_entropy([1, 0], [1.1, -0.1])
    with pytest.raises(ValueError, match="batch size"):
        categorical_cross_entropy_batch([[1, 0]], [[1, 0], [0, 1]])
