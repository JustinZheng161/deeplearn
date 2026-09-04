import random

import pytest

from deeplearn_utils import derive_seed, make_worker_init_fn, seed_everything


def test_seed_everything_repeats_python_and_numpy_sequences():
    first_report = seed_everything(7)
    first_values = [random.random() for _ in range(3)]

    second_report = seed_everything(7)
    second_values = [random.random() for _ in range(3)]

    assert first_values == second_values
    assert first_report == second_report
    assert first_report.seed == 7


def test_seed_everything_rejects_bool_and_negative_values():
    with pytest.raises(TypeError):
        seed_everything(True)
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_worker_initializer_gives_workers_distinct_repeatable_streams():
    initialize = make_worker_init_fn(100)
    initialize(0)
    worker_zero = random.random()
    initialize(1)
    worker_one = random.random()

    initialize(0)
    assert random.random() == worker_zero
    assert worker_zero != worker_one


def test_worker_initializer_rejects_invalid_worker_id():
    with pytest.raises(ValueError):
        make_worker_init_fn(1)(-1)


def test_derived_stream_seeds_are_stable_and_distinct():
    assert derive_seed(42, 0) == derive_seed(42, 0)
    assert derive_seed(42, 0) != derive_seed(42, 1)
    assert 0 <= derive_seed(42, 0) < 2**32


@pytest.mark.parametrize(
    ("seed", "stream", "error"),
    [(True, 0, TypeError), (-1, 0, ValueError), (1, True, TypeError), (1, -1, ValueError)],
)
def test_derived_stream_seeds_validate_inputs(seed, stream, error):
    with pytest.raises(error):
        derive_seed(seed, stream)
