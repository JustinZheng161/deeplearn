import pytest

from deeplearn_utils import mean_absolute_error, mean_squared_error, r2_score


def test_regression_metrics_match_reference_values():
    targets = [1.0, 2.0, 4.0]
    predictions = [1.5, 1.0, 5.0]

    assert mean_absolute_error(targets, predictions) == pytest.approx(5 / 6)
    assert mean_squared_error(targets, predictions) == pytest.approx(0.75)
    assert r2_score(targets, predictions) == pytest.approx(29 / 56)


def test_r2_handles_constant_targets_without_nan():
    assert r2_score([3, 3], [3, 3]) == 1.0
    assert r2_score([3, 3], [2, 3]) == 0.0


@pytest.mark.parametrize("metric", [mean_absolute_error, mean_squared_error, r2_score])
def test_regression_metrics_validate_empty_and_mismatched_inputs(metric):
    with pytest.raises(ValueError, match="must not be empty"):
        metric([], [])
    with pytest.raises(ValueError, match="same length"):
        metric([1], [1, 2])


def test_regression_metrics_reject_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        mean_squared_error([1.0], [float("nan")])
    with pytest.raises(TypeError, match="only numbers"):
        mean_absolute_error([1.0], ["1"])
