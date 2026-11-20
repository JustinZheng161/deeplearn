import math

import pytest

from deeplearn_utils import (
    log_cosh_loss,
    log_cosh_loss_batch,
    mean_absolute_error,
    mean_squared_error,
    quantile_loss,
    quantile_loss_batch,
    r2_score,
)


def test_regression_metrics_match_reference_values():
    targets = [1.0, 2.0, 4.0]
    predictions = [1.5, 1.0, 5.0]

    assert mean_absolute_error(targets, predictions) == pytest.approx(5 / 6)
    assert mean_squared_error(targets, predictions) == pytest.approx(0.75)
    assert r2_score(targets, predictions) == pytest.approx(29 / 56)


def test_log_cosh_loss_is_smooth_and_stable_for_large_errors():
    assert log_cosh_loss(1.0, 1.0) == pytest.approx(0.0)
    assert log_cosh_loss(0.0, 1.0) == pytest.approx(math.log(math.cosh(1.0)))
    assert math.isfinite(log_cosh_loss(0.0, 1000.0))
    assert log_cosh_loss(0.0, 2.0) < mean_squared_error([0.0], [2.0])


def test_log_cosh_loss_batch_returns_mean():
    assert log_cosh_loss_batch([0.0, 0.0], [1.0, 2.0]) == pytest.approx(
        (log_cosh_loss(0.0, 1.0) + log_cosh_loss(0.0, 2.0)) / 2
    )


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


def test_log_cosh_loss_validates_scalar_inputs():
    with pytest.raises(TypeError):
        log_cosh_loss(1.0, "1")
    with pytest.raises(ValueError):
        log_cosh_loss(float("nan"), 1.0)
    with pytest.raises(ValueError):
        log_cosh_loss_batch([], [])


def test_huber_loss_uses_quadratic_and_linear_regions():
    from deeplearn_utils import huber_loss

    assert huber_loss(3.0, 2.5, delta=1.0) == pytest.approx(0.125)
    assert huber_loss(4.0, 1.0, delta=1.0) == pytest.approx(2.5)
    assert huber_loss(1.0, 1.0) == 0.0


def test_huber_loss_batch_returns_mean():
    from deeplearn_utils import huber_loss_batch

    assert huber_loss_batch([0, 3], [1, 1], delta=1.0) == pytest.approx(1.0)


def test_quantile_loss_weights_under_and_over_predictions():
    assert quantile_loss(4.0, 3.0, quantile=0.8) == pytest.approx(0.8)
    assert quantile_loss(3.0, 4.0, quantile=0.8) == pytest.approx(0.2)
    assert quantile_loss(2.0, 2.0, quantile=0.5) == 0.0


def test_quantile_loss_batch_supports_scalar_and_multiple_quantiles():
    assert quantile_loss_batch([1.0, 3.0], [2.0, 2.0], quantile=0.5) == pytest.approx(0.5)
    result = quantile_loss_batch([2.0], [[1.0, 3.0]], quantile=[0.25, 0.75])
    assert result == pytest.approx(0.25)


@pytest.mark.parametrize("level", [0, 1, -0.1, 1.1, float("nan"), True, "0.5"])
def test_quantile_loss_validates_quantile(level):
    with pytest.raises((TypeError, ValueError)):
        quantile_loss(1.0, 2.0, quantile=level)


def test_quantile_loss_batch_validates_rows_and_quantile_count():
    with pytest.raises(ValueError, match="same length"):
        quantile_loss_batch([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="quantile count"):
        quantile_loss_batch([1.0], [[1.0, 2.0]], quantile=[0.5])


@pytest.mark.parametrize("delta", [0, -1, float("inf"), True, "1"])
def test_huber_loss_validates_delta(delta):
    from deeplearn_utils import huber_loss

    with pytest.raises((TypeError, ValueError)):
        huber_loss(1, 2, delta=delta)
