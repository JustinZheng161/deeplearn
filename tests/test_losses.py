import math

import pytest

from deeplearn_utils import binary_focal_loss, dice_coefficient, dice_loss, multiclass_focal_loss


def test_binary_focal_loss_is_lower_for_confident_correct_predictions():
    confident = binary_focal_loss([1, 0], [0.99, 0.01])
    uncertain = binary_focal_loss([1, 0], [0.5, 0.5])
    assert confident < uncertain


def test_binary_focal_loss_supports_class_weighting():
    positive = binary_focal_loss([1], [0.8], alpha=0.75)
    negative = binary_focal_loss([0], [0.2], alpha=0.75)
    assert positive != negative


def test_multiclass_focal_loss_validates_normalized_rows():
    loss = multiclass_focal_loss([0, 1], [[0.9, 0.1], [0.2, 0.8]])
    assert loss > 0
    assert loss < multiclass_focal_loss([0, 1], [[0.5, 0.5], [0.5, 0.5]])


def test_dice_coefficient_and_loss_are_complements():
    coefficient = dice_coefficient([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2])
    assert 0 < coefficient <= 1
    assert dice_loss([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2]) == pytest.approx(1 - coefficient)


def test_dice_handles_perfect_and_empty_masks_with_smoothing():
    assert dice_coefficient([1, 0], [1, 0], smooth=0) == pytest.approx(1.0)
    assert dice_coefficient([0, 0], [0, 0], smooth=1) == pytest.approx(1.0)


@pytest.mark.parametrize("gamma", [-1, float("nan"), True, "2"])
def test_binary_focal_validates_gamma(gamma):
    with pytest.raises((TypeError, ValueError)):
        binary_focal_loss([1], [0.5], gamma=gamma)


def test_binary_focal_validates_labels_and_probabilities():
    with pytest.raises(ValueError):
        binary_focal_loss([2], [0.5])
    with pytest.raises(ValueError):
        binary_focal_loss([1], [1.5])
    with pytest.raises(ValueError):
        binary_focal_loss([1], [])


def test_multiclass_focal_validates_shapes_and_probability_sums():
    with pytest.raises(ValueError):
        multiclass_focal_loss([0], [[0.2, 0.2]])
    with pytest.raises(ValueError):
        multiclass_focal_loss([2], [[0.5, 0.5]])
    with pytest.raises(ValueError):
        multiclass_focal_loss([0, 1], [[1.0, 0.0], [1.0]])


def test_dice_validates_values_and_shapes():
    with pytest.raises(ValueError):
        dice_coefficient([1], [0.5, 0.5])
    with pytest.raises(ValueError):
        dice_coefficient([2], [0.5])
    with pytest.raises(ValueError):
        dice_coefficient([1], [0.5], smooth=-1)


def test_focal_loss_is_finite_at_probability_boundaries():
    assert math.isfinite(binary_focal_loss([1, 0], [1.0, 0.0]))
    assert math.isfinite(multiclass_focal_loss([0], [[1.0, 0.0]]))


def test_tversky_index_weights_false_negatives_and_complements_loss():
    from deeplearn_utils import tversky_index, tversky_loss

    index = tversky_index([1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1], alpha=0.3, beta=0.7)
    assert 0.0 < index < 1.0
    assert tversky_loss([1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1], alpha=0.3, beta=0.7) == pytest.approx(1 - index)
    assert tversky_index([1, 0], [1, 0], smooth=0) == pytest.approx(1.0)


@pytest.mark.parametrize("alpha,beta", [(0, 0), (-1, 1), (1, -1), (True, 1)])
def test_tversky_validates_error_weights(alpha, beta):
    from deeplearn_utils import tversky_index

    with pytest.raises((TypeError, ValueError)):
        tversky_index([1], [0.5], alpha=alpha, beta=beta)
