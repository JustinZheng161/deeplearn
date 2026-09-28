import pytest

from deeplearn_utils import (
    TemperatureScaler,
    calibrated_probabilities,
    negative_log_likelihood,
    reliability_bins,
)


def test_calibrated_probabilities_are_normalized_and_temperature_softens_logits():
    sharp = calibrated_probabilities([[4.0, 0.0]], temperature=1.0)[0]
    soft = calibrated_probabilities([[4.0, 0.0]], temperature=2.0)[0]
    assert sum(sharp) == pytest.approx(1.0)
    assert soft[0] < sharp[0]


def test_negative_log_likelihood_prefers_correct_class_logits():
    good = negative_log_likelihood([[4.0, 0.0], [0.0, 4.0]], [0, 1])
    bad = negative_log_likelihood([[0.0, 4.0], [4.0, 0.0]], [0, 1])
    assert good < bad


def test_temperature_scaler_fit_is_deterministic_and_transforms():
    logits = [[4.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 2.0]]
    labels = [0, 0, 1, 1]
    first = TemperatureScaler().fit(logits, labels, steps=8)
    second = TemperatureScaler().fit(logits, labels, steps=8)
    assert first.temperature == pytest.approx(second.temperature)
    assert first.fitted
    assert len(first.transform(logits)) == 4


def test_reliability_bins_have_valid_ranges_and_counts():
    bins = reliability_bins([[4.0, 0.0], [0.0, 4.0], [1.0, 1.0]], [0, 1, 0], bins=2)
    assert sum(item.count for item in bins) == 3
    assert all(0 <= item.lower < item.upper <= 1 for item in bins)
    assert all(0 <= item.accuracy <= 1 and 0 <= item.confidence <= 1 for item in bins)


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), True, "1"])
def test_probability_helpers_validate_temperature(temperature):
    with pytest.raises((TypeError, ValueError)):
        calibrated_probabilities([[1, 0]], temperature)


def test_calibration_validates_shapes_and_labels():
    with pytest.raises(ValueError):
        negative_log_likelihood([[1, 0]], [0, 1])
    with pytest.raises(ValueError):
        calibrated_probabilities([[1]])
    with pytest.raises(ValueError):
        negative_log_likelihood([[1, 0]], [2])


def test_reliability_bins_validate_bin_count():
    with pytest.raises(ValueError):
        reliability_bins([[1, 0]], [0], bins=0)
    with pytest.raises(TypeError):
        reliability_bins([[1, 0]], [0], bins=True)


def test_temperature_scaler_validates_fit_steps():
    scaler = TemperatureScaler()
    with pytest.raises(ValueError):
        scaler.fit([[1, 0]], [0], steps=0)
    with pytest.raises(TypeError):
        scaler.fit([[1, 0]], [0], steps=True)
