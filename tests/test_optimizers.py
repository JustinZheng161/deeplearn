import pytest

from deeplearn_utils import Adam, SGD


def test_sgd_without_momentum_matches_gradient_descent():
    optimizer = SGD(learning_rate=0.1)
    assert optimizer.step([1.0, -2.0], [0.5, -1.0]) == pytest.approx([0.95, -1.9])


def test_sgd_momentum_accumulates_state_and_resets():
    optimizer = SGD(learning_rate=0.1, momentum=0.9)
    first = optimizer.step([1.0], [1.0])
    second = optimizer.step(first, [1.0])
    assert second[0] < first[0] - 0.09
    assert optimizer.velocity
    optimizer.reset()
    assert optimizer.velocity == []


def test_sgd_weight_decay_and_nesterov_are_supported():
    optimizer = SGD(learning_rate=0.1, momentum=0.9, weight_decay=0.1, nesterov=True)
    result = optimizer.step([1.0], [0.0])
    assert result[0] < 1.0


def test_adam_updates_and_tracks_bias_correction_state():
    optimizer = Adam(learning_rate=0.1)
    first = optimizer.step([1.0, -1.0], [1.0, -1.0])
    second = optimizer.step(first, [1.0, -1.0])
    assert first == pytest.approx([0.9, -0.9], abs=1e-6)
    assert second[0] < first[0]
    assert optimizer.step_count == 2
    assert len(optimizer.first_moment) == 2


def test_adam_reset_clears_moments():
    optimizer = Adam()
    optimizer.step([1.0], [1.0])
    optimizer.reset()
    assert optimizer.step_count == 0
    assert optimizer.first_moment == []
    assert optimizer.second_moment == []


def test_adam_decoupled_weight_decay_changes_parameter_even_with_zero_gradient():
    optimizer = Adam(learning_rate=0.1, weight_decay=0.2)
    assert optimizer.step([1.0], [0.0])[0] == pytest.approx(0.98)


def test_optimizers_validate_lengths_and_dimension_changes():
    optimizer = SGD()
    with pytest.raises(ValueError, match="same length"):
        optimizer.step([1], [1, 2])
    optimizer.step([1], [1])
    with pytest.raises(ValueError, match="dimension"):
        optimizer.step([1, 2], [1, 2])


def test_optimizers_validate_hyperparameters():
    with pytest.raises(ValueError):
        SGD(learning_rate=0)
    with pytest.raises(ValueError):
        SGD(momentum=1)
    with pytest.raises(ValueError):
        SGD(nesterov=True)
    with pytest.raises(ValueError):
        Adam(beta1=1)
    with pytest.raises(ValueError):
        Adam(epsilon=0)


def test_optimizers_reject_invalid_values():
    with pytest.raises(ValueError):
        SGD().step([float("inf")], [1])
    with pytest.raises(TypeError):
        Adam().step([True], [1])
