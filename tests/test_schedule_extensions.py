import math

import pytest

from deeplearn_utils import cosine_restart, inverse_sqrt_decay, polynomial_decay


def test_polynomial_decay_reaches_final_rate_and_is_monotonic():
    values = [polynomial_decay(step, total_steps=4, initial_lr=1.0, final_lr=0.1, power=2) for step in range(6)]
    assert values[0] == pytest.approx(1.0)
    assert values[4] == pytest.approx(0.1)
    assert values[5] == pytest.approx(0.1)
    assert values == sorted(values, reverse=True)


def test_inverse_sqrt_decay_warms_up_then_decays():
    warmup = [inverse_sqrt_decay(step, warmup_steps=4, peak_lr=1.0) for step in range(4)]
    assert warmup == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert inverse_sqrt_decay(7, warmup_steps=4, peak_lr=1.0) == pytest.approx(math.sqrt(0.5))


def test_inverse_sqrt_respects_minimum_rate():
    assert inverse_sqrt_decay(10_000, warmup_steps=2, peak_lr=1.0, min_lr=0.2) == pytest.approx(0.2)


def test_cosine_restart_returns_peaks_at_restart_boundaries():
    values = [cosine_restart(step, first_period=4, peak_lr=1.0, min_lr=0.1) for step in range(8)]
    assert values[0] == pytest.approx(1.0)
    assert values[3] == pytest.approx(0.1, abs=1e-3)
    assert values[4] == pytest.approx(1.0)


def test_cosine_restart_supports_period_multiplier():
    assert cosine_restart(0, first_period=2, peak_lr=1.0, period_multiplier=2) == pytest.approx(1.0)
    assert cosine_restart(2, first_period=2, peak_lr=1.0, period_multiplier=2) == pytest.approx(1.0)


@pytest.mark.parametrize("function", [polynomial_decay, inverse_sqrt_decay, cosine_restart])
def test_schedule_extensions_reject_negative_steps(function):
    with pytest.raises(ValueError):
        if function is polynomial_decay:
            function(-1, total_steps=2, initial_lr=1)
        elif function is inverse_sqrt_decay:
            function(-1, warmup_steps=2, peak_lr=1)
        else:
            function(-1, first_period=2, peak_lr=1)


def test_polynomial_decay_validates_ordered_parameters():
    with pytest.raises(ValueError):
        polynomial_decay(0, total_steps=1, initial_lr=1, final_lr=2)
    with pytest.raises(ValueError):
        polynomial_decay(0, total_steps=1, initial_lr=1, power=0)
