import pytest

from deeplearn_utils import expected_calibration_error


def test_expected_calibration_error_is_zero_for_perfectly_calibrated_bins():
    correct = [True, False, True, False]
    confidences = [1.0, 0.0, 1.0, 0.0]

    assert expected_calibration_error(correct, confidences, n_bins=2) == pytest.approx(0.0)


def test_expected_calibration_error_weights_bins_by_sample_count():
    correct = [True, True, False]
    confidences = [0.9, 0.9, 0.1]

    assert expected_calibration_error(correct, confidences, n_bins=2) == pytest.approx(0.1)


def test_confidence_one_is_included_in_the_last_bin():
    assert expected_calibration_error([True], [1.0], n_bins=10) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("correct", "confidences", "error"),
    [([True], [], "same length"), ([1], [0.5], "booleans"), ([True], [1.1], "between")],
)
def test_expected_calibration_error_validates_inputs(correct, confidences, error):
    with pytest.raises((TypeError, ValueError), match=error):
        expected_calibration_error(correct, confidences)


def test_expected_calibration_error_validates_bin_count():
    with pytest.raises(ValueError, match="positive"):
        expected_calibration_error([True], [0.5], n_bins=0)
    with pytest.raises(TypeError, match="integer"):
        expected_calibration_error([True], [0.5], n_bins=1.5)
