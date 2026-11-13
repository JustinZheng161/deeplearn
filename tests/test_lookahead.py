import pytest

from deeplearn_utils import Adam, Lookahead, SGD


def test_lookahead_delays_and_then_interpolates_slow_weights():
    optimizer = Lookahead(SGD(learning_rate=0.1), alpha=0.5, sync_period=2)
    first = optimizer.step([1.0], [1.0])
    second = optimizer.step(first, [1.0])
    assert first == pytest.approx([0.9])
    assert second == pytest.approx([0.9])
    assert optimizer.synchronized is True
    assert optimizer.slow_parameters == pytest.approx([0.9])


def test_lookahead_wraps_adam_and_preserves_optimizer_state():
    optimizer = Lookahead(Adam(learning_rate=0.1), sync_period=3)
    values = [1.0, -1.0]
    for _ in range(3):
        values = optimizer.step(values, [0.5, -0.5])
    assert values[0] < 1.0 and values[1] > -1.0
    assert optimizer.step_count == 3
    assert optimizer.optimizer.step_count == 3


def test_lookahead_reset_clears_wrapper_and_inner_state():
    optimizer = Lookahead(SGD(momentum=0.9), sync_period=2)
    optimizer.step([1.0], [1.0])
    optimizer.reset()
    assert optimizer.step_count == 0
    assert optimizer.slow_parameters == []
    assert optimizer.optimizer.velocity == []


def test_lookahead_validates_configuration_and_dimensions():
    with pytest.raises(TypeError):
        Lookahead(object())
    with pytest.raises(ValueError):
        Lookahead(SGD(), alpha=0)
    with pytest.raises(ValueError):
        Lookahead(SGD(), sync_period=0)
    optimizer = Lookahead(SGD())
    optimizer.step([1.0], [1.0])
    with pytest.raises(ValueError):
        optimizer.step([1.0, 2.0], [1.0, 1.0])
