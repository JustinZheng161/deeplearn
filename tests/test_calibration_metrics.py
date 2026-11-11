import pytest

from deeplearn_utils import adaptive_calibration_error, classwise_calibration_error


def test_adaptive_calibration_error_is_zero_for_perfect_confidence():
    assert adaptive_calibration_error([True, False, True, False], [1.0, 0.0, 1.0, 0.0], n_bins=2) == pytest.approx(0.0)


def test_adaptive_calibration_error_uses_equal_count_bins():
    score = adaptive_calibration_error([True, False, True, False], [0.1, 0.2, 0.8, 0.9], n_bins=2)
    assert score == pytest.approx(0.35)
    assert adaptive_calibration_error([True], [0.7], n_bins=10) == pytest.approx(0.3)


def test_classwise_calibration_error_preserves_class_order():
    labels = [0, 1, 2, 1]
    probabilities = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
    assert classwise_calibration_error(labels, probabilities, n_bins=2) == pytest.approx([0.0, 0.0, 0.0])


def test_classwise_calibration_error_rejects_malformed_rows():
    with pytest.raises(ValueError):
        classwise_calibration_error([0], [[0.5]])
    with pytest.raises(ValueError):
        classwise_calibration_error([1], [[0.8, 0.1]])
    with pytest.raises(ValueError):
        classwise_calibration_error([0, 1], [[0.5, 0.5]])


def test_calibration_metrics_validate_bin_count_and_confidences():
    with pytest.raises(ValueError):
        adaptive_calibration_error([True], [1.2])
    with pytest.raises(ValueError):
        adaptive_calibration_error([True], [0.5], n_bins=0)
    with pytest.raises(TypeError):
        adaptive_calibration_error([1], [0.5])
