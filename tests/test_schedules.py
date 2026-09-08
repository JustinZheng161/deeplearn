import pytest

from deeplearn_utils import warmup_cosine_decay


def test_warmup_reaches_peak_learning_rate():
    values = [
        warmup_cosine_decay(
            step, warmup_steps=2, total_steps=6, peak_lr=1.0, min_lr=0.1
        )
        for step in range(2)
    ]

    assert values == pytest.approx([0.5, 1.0])


def test_cosine_decay_reaches_minimum_and_clamps_after_training():
    values = [
        warmup_cosine_decay(
            step, warmup_steps=0, total_steps=4, peak_lr=1.0, min_lr=0.2
        )
        for step in range(5)
    ]

    assert values[0] == pytest.approx(1.0)
    assert values[-2:] == pytest.approx([0.317157, 0.2], abs=1e-5)
    assert warmup_cosine_decay(
        100, warmup_steps=0, total_steps=4, peak_lr=1.0, min_lr=0.2
    ) == pytest.approx(0.2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step": -1},
        {"warmup_steps": 6},
        {"total_steps": 0},
        {"peak_lr": 0},
        {"min_lr": 1.1},
    ],
)
def test_schedule_rejects_invalid_values(kwargs):
    params = {
        "step": 0,
        "warmup_steps": 2,
        "total_steps": 6,
        "peak_lr": 1.0,
        "min_lr": 0.1,
    }
    params.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        warmup_cosine_decay(**params)


def test_schedule_rejects_boolean_integer_arguments():
    with pytest.raises(TypeError, match="step"):
        warmup_cosine_decay(True, warmup_steps=0, total_steps=2, peak_lr=1.0)
