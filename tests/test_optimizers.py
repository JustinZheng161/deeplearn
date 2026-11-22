import copy

import pytest

from deeplearn_utils import AdaGrad, Adam, Lion, RAdam, RMSprop, SAM, SGD


def test_sam_applies_gradient_perturbation_then_second_update():
    optimizer = SAM(learning_rate=0.1, rho=0.1)
    perturbed = optimizer.first_step([3.0, 4.0], [3.0, 4.0])
    assert perturbed == pytest.approx([3.06, 4.08])
    assert optimizer.second_step([3.06, 4.08], [1.0, -2.0]) == pytest.approx([2.9, 4.2])
    assert optimizer.perturbation == []


def test_sam_supports_adaptive_and_zero_gradient_paths():
    optimizer = SAM(learning_rate=0.1, rho=0.2, adaptive=True)
    assert optimizer.first_step([2.0, 3.0], [0.0, 0.0]) == [2.0, 3.0]
    optimizer.reset()
    assert optimizer.perturbation == []
    with pytest.raises(RuntimeError, match="second_step"):
        optimizer.second_step([1.0], [1.0])
    optimizer.first_step([1.0], [1.0])
    with pytest.raises(RuntimeError, match="first_step"):
        optimizer.first_step([1.0], [1.0])


def test_sam_validates_configuration_and_step_dimensions():
    with pytest.raises(ValueError):
        SAM(rho=0)
    with pytest.raises(TypeError):
        SAM(adaptive=1)
    optimizer = SAM()
    with pytest.raises(ValueError, match="same length"):
        optimizer.first_step([1.0], [1.0, 2.0])
    optimizer.first_step([1.0], [1.0])
    with pytest.raises(ValueError, match="dimension"):
        optimizer.second_step([1.0, 2.0], [1.0, 2.0])


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


def test_lion_uses_sign_updates_and_tracks_momentum():
    optimizer = Lion(learning_rate=0.1, beta1=0.5, beta2=0.9)
    first = optimizer.step([1.0, -1.0], [2.0, -2.0])
    second = optimizer.step(first, [2.0, -2.0])
    assert first == pytest.approx([0.9, -0.9])
    assert second == pytest.approx([0.8, -0.8])
    assert optimizer.momentum == pytest.approx([0.38, -0.38])


def test_lion_decoupled_decay_and_reset():
    optimizer = Lion(learning_rate=0.1, weight_decay=0.2)
    assert optimizer.step([1.0], [0.0])[0] == pytest.approx(0.98)
    optimizer.reset()
    assert optimizer.momentum == []


def test_radam_rectifies_late_updates_and_tracks_state():
    optimizer = RAdam(learning_rate=0.1, beta1=0.9, beta2=0.9)
    first = optimizer.step([1.0], [1.0])
    assert first[0] == pytest.approx(0.9)
    for _ in range(10):
        first = optimizer.step(first, [1.0])
    assert first[0] < 0.9
    assert optimizer.step_count == 11
    assert optimizer.first_moment and optimizer.second_moment


def test_radam_supports_decoupled_decay_and_reset():
    optimizer = RAdam(learning_rate=0.1, beta2=0.9, weight_decay=0.2)
    result = optimizer.step([1.0], [0.0])
    assert result[0] == pytest.approx(0.98)
    optimizer.reset()
    assert optimizer.step_count == 0
    assert optimizer.first_moment == []
    assert optimizer.second_moment == []


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
        AdaGrad(learning_rate=0)
    with pytest.raises(ValueError):
        AdaGrad(epsilon=0)
    with pytest.raises((TypeError, ValueError)):
        AdaGrad(weight_decay=-1)
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
    with pytest.raises(ValueError):
        Lion(beta2=1)
    with pytest.raises(ValueError):
        Lion(weight_decay=-1)
    with pytest.raises(ValueError):
        RAdam(beta2=1)
    with pytest.raises(ValueError):
        RAdam(epsilon=0)


def test_optimizers_reject_invalid_values():
    with pytest.raises(ValueError):
        SGD().step([float("inf")], [1])
    with pytest.raises(TypeError):
        Adam().step([True], [1])


def test_adagrad_accumulates_squared_gradients_and_adapts_steps():
    optimizer = AdaGrad(learning_rate=0.1, epsilon=1e-8)
    first = optimizer.step([1.0], [1.0])
    second = optimizer.step(first, [1.0])
    assert first[0] == pytest.approx(0.9, abs=1e-6)
    assert second[0] == pytest.approx(0.829289, abs=1e-5)
    assert optimizer.squared_gradient == pytest.approx([2.0])


def test_adagrad_supports_weight_decay_and_reset():
    optimizer = AdaGrad(learning_rate=0.1, weight_decay=0.1)
    result = optimizer.step([1.0], [0.0])
    assert result[0] < 1.0
    optimizer.reset()
    assert optimizer.squared_gradient == []


def test_rmsprop_adapts_updates_and_tracks_state():
    from deeplearn_utils import RMSprop

    optimizer = RMSprop(learning_rate=0.1, alpha=0.9, momentum=0.5, centered=True)
    first = optimizer.step([1.0], [1.0])
    second = optimizer.step(first, [1.0])
    assert first[0] < 1.0
    assert second[0] < first[0]
    assert optimizer.square_average and optimizer.momentum_buffer and optimizer.gradient_average


def test_rmsprop_reset_clears_running_statistics():
    from deeplearn_utils import RMSprop

    optimizer = RMSprop()
    optimizer.step([1.0], [1.0])
    optimizer.reset()
    assert optimizer.square_average == []
    assert optimizer.momentum_buffer == []
    assert optimizer.gradient_average == []


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SGD(learning_rate=0.1, momentum=0.8),
        lambda: Adam(learning_rate=0.1),
        lambda: Lion(learning_rate=0.1),
        lambda: RAdam(learning_rate=0.1),
        lambda: AdaGrad(learning_rate=0.1),
        lambda: RMSprop(learning_rate=0.1, centered=True, momentum=0.2),
    ],
)
def test_optimizer_state_dict_round_trips_for_continuation(factory):
    original = factory()
    parameters = original.step([1.0, -1.0], [0.5, -0.5])
    checkpoint = original.state_dict()
    restored = factory()
    restored.load_state_dict(checkpoint)
    checkpoint["step_count"] = 999 if "step_count" in checkpoint else checkpoint.get("step_count")
    assert restored.state_dict() != checkpoint
    assert restored.step(parameters, [0.25, -0.25]) == pytest.approx(
        original.step(parameters, [0.25, -0.25])
    )


def test_optimizer_state_dict_is_independent_and_rejects_bad_keys():
    optimizer = Adam()
    optimizer.step([1.0], [1.0])
    state = optimizer.state_dict()
    copied = copy.deepcopy(state)
    copied["first_moment"][0] = 123.0
    assert optimizer.first_moment[0] != 123.0
    with pytest.raises(ValueError, match="state keys"):
        optimizer.load_state_dict({"unexpected": 1})


def test_invalid_optimizer_state_is_rejected_without_mutating_configuration():
    optimizer = SGD(learning_rate=0.2)
    state = optimizer.state_dict()
    state["learning_rate"] = 0
    with pytest.raises(ValueError):
        optimizer.load_state_dict(state)
    assert optimizer.learning_rate == pytest.approx(0.2)


@pytest.mark.parametrize("kwargs", [{"alpha": 1}, {"momentum": -1}, {"epsilon": 0}, {"centered": 1}])
def test_rmsprop_validates_hyperparameters(kwargs):
    from deeplearn_utils import RMSprop

    with pytest.raises((TypeError, ValueError)):
        RMSprop(**kwargs)
