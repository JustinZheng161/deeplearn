import random

import pytest

from deeplearn_utils import (
    EarlyStopping,
    RunningAverage,
    batch_indices,
    iter_minibatches,
)


def test_batch_indices_cover_items_and_keep_final_short_batch():
    batches = list(batch_indices(7, 3))
    assert batches == [[0, 1, 2], [3, 4, 5], [6]]
    assert sorted(index for batch in batches for index in batch) == list(range(7))


def test_batch_indices_shuffle_is_reproducible_without_touching_global_rng():
    random.seed(9)
    before = random.random()
    first = list(batch_indices(10, 4, shuffle=True, seed=7))
    after = random.random()
    random.seed(9)
    assert before == random.random()
    assert after == random.random()
    assert first == list(batch_indices(10, 4, shuffle=True, seed=7))
    assert first != list(batch_indices(10, 4, shuffle=True, seed=8))


@pytest.mark.parametrize(
    ("size", "batch_size", "error"),
    [(True, 2, TypeError), (-1, 2, ValueError), (3, 0, ValueError), (3, 1.5, TypeError)],
)
def test_batch_indices_validates_sizes(size, batch_size, error):
    with pytest.raises(error):
        list(batch_indices(size, batch_size))


def test_iter_minibatches_keeps_features_and_targets_aligned():
    result = list(iter_minibatches(["a", "b", "c"], [1, 2, 3], 2))
    assert result == [(["a", "b"], [1, 2]), (["c"], [3])]


def test_iter_minibatches_shuffles_pairs_together():
    batches = list(iter_minibatches(list(range(6)), list("abcdef"), 2, shuffle=True, seed=4))
    assert sorted(zip(sum((x for x, _ in batches), []), sum((y for _, y in batches), []))) == list(
        zip(range(6), "abcdef")
    )


def test_iter_minibatches_rejects_mismatched_sequences():
    with pytest.raises(ValueError, match="same length"):
        list(iter_minibatches([1], [1, 2], 1))


def test_running_average_weights_batches_and_resets():
    average = RunningAverage()
    assert average.mean == 0.0
    average.update(2.0, weight=2)
    average.update(5.0, weight=1)
    assert average.mean == pytest.approx(3.0)
    average.reset()
    assert average.total == 0.0
    assert average.weight == 0


@pytest.mark.parametrize(
    ("value", "weight", "error"),
    [("loss", 1, TypeError), (1.0, 0, ValueError), (1.0, 1.5, TypeError)],
)
def test_running_average_validates_updates(value, weight, error):
    with pytest.raises(error):
        RunningAverage().update(value, weight=weight)


def test_early_stopping_min_mode_stops_after_patience_bad_epochs():
    stopping = EarlyStopping(patience=2, mode="min")
    assert stopping.update(1.0) is False
    assert stopping.update(1.1) is False
    assert stopping.update(1.2) is True
    assert stopping.state().stopped is True
    stopping.reset()
    assert stopping.state().best is None


def test_early_stopping_max_mode_and_min_delta():
    stopping = EarlyStopping(patience=2, mode="max", min_delta=0.1)
    assert stopping.update(0.5) is False
    assert stopping.update(0.55) is False
    assert stopping.update(0.7) is False
    assert stopping.state().best == 0.7


@pytest.mark.parametrize(
    "kwargs",
    [{"patience": 0}, {"mode": "sideways"}, {"min_delta": -1}, {"patience": True}],
)
def test_early_stopping_validates_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        EarlyStopping(**kwargs)
