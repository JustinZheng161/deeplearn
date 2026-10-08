import pytest

from deeplearn_utils import (
    clip_vector_norm,
    elastic_net_penalty,
    l1_penalty,
    l2_penalty,
    weight_decay_update,
)


def test_l1_and_l2_penalties_support_nested_parameters():
    parameters = [[1, -2], [3]]
    assert l1_penalty(parameters) == pytest.approx(6.0)
    assert l2_penalty(parameters) == pytest.approx(14.0)
    assert l1_penalty(parameters, coefficient=0.5) == pytest.approx(3.0)


def test_elastic_net_combines_both_penalties():
    assert elastic_net_penalty([1, -2], l1_coefficient=0.5, l2_coefficient=0.25) == pytest.approx(2.75)


def test_weight_decay_update_applies_decay_and_gradient_step():
    result = weight_decay_update([1.0, -2.0], [0.5, -1.0], learning_rate=0.1, decay=0.01)
    assert result == pytest.approx([0.949, -1.898])


def test_clip_vector_norm_preserves_direction_and_limit():
    assert clip_vector_norm([3, 4], max_norm=10) == pytest.approx([3, 4])
    assert clip_vector_norm([3, 4], max_norm=1) == pytest.approx([0.6, 0.8])
    assert clip_vector_norm([0, 0], max_norm=1) == [0.0, 0.0]


def test_regularizers_do_not_mutate_inputs():
    parameters = [[1.0, 2.0]]
    original = [row[:] for row in parameters]
    l1_penalty(parameters)
    clip_vector_norm(parameters[0], max_norm=1)
    assert parameters == original


@pytest.mark.parametrize("coefficient", [-1, float("nan"), True, "1"])
def test_penalties_validate_coefficients(coefficient):
    with pytest.raises((TypeError, ValueError)):
        l1_penalty([1], coefficient=coefficient)


def test_elastic_net_validates_each_coefficient():
    with pytest.raises(ValueError):
        elastic_net_penalty([1], l1_coefficient=-1)
    with pytest.raises(TypeError):
        elastic_net_penalty([1], l2_coefficient=True)


def test_weight_decay_validates_lengths_and_hyperparameters():
    with pytest.raises(ValueError, match="same length"):
        weight_decay_update([1], [1, 2], learning_rate=0.1, decay=0.1)
    with pytest.raises(ValueError):
        weight_decay_update([1], [1], learning_rate=0, decay=0.1)
    with pytest.raises(ValueError):
        weight_decay_update([1], [1], learning_rate=0.1, decay=-1)


def test_clip_vector_norm_validates_inputs():
    with pytest.raises(ValueError):
        clip_vector_norm([], max_norm=1)
    with pytest.raises(ValueError):
        clip_vector_norm([1], max_norm=0)
    with pytest.raises(TypeError):
        clip_vector_norm([True], max_norm=1)


def test_nested_parameter_validation_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        l2_penalty([[1, float("inf")]])
    with pytest.raises(TypeError):
        l2_penalty([[1, "bad"]])


def test_group_lasso_sums_group_norms_and_supports_coefficients():
    from deeplearn_utils import group_lasso_penalty

    assert group_lasso_penalty([[3, 4], [0, 6]]) == pytest.approx(11.0)
    assert group_lasso_penalty([[3, 4], [0, 6]], coefficient=0.5) == pytest.approx(5.5)


@pytest.mark.parametrize("groups", [[], "abc"])
def test_group_lasso_validates_groups(groups):
    from deeplearn_utils import group_lasso_penalty

    with pytest.raises((TypeError, ValueError)):
        group_lasso_penalty(groups)
