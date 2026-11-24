import math

import pytest

from deeplearn_utils import (
    ensemble_mean,
    ensemble_regression_mean,
    ensemble_regression_variance,
    entropy,
    mutual_information,
    predictive_entropy,
    summarize_ensemble,
    variation_ratio,
)


MEMBERS = [
    [[0.9, 0.1], [0.6, 0.4]],
    [[0.8, 0.2], [0.4, 0.6]],
    [[0.7, 0.3], [0.5, 0.5]],
]


def test_entropy_supports_nats_and_bits():
    assert entropy([0.5, 0.5]) == pytest.approx(math.log(2))
    assert entropy([0.5, 0.5], base=2) == pytest.approx(1.0)
    assert entropy([1.0, 0.0]) == pytest.approx(0.0)


def test_ensemble_mean_preserves_sample_and_class_shape():
    result = ensemble_mean(MEMBERS)
    assert result[0] == pytest.approx([0.8, 0.2])
    assert result[1] == pytest.approx([0.5, 0.5])


def test_predictive_entropy_matches_ensemble_mean():
    mean = ensemble_mean(MEMBERS)[0]
    assert predictive_entropy(mean) == pytest.approx(entropy(mean))


def test_mutual_information_is_nonnegative_for_ensemble():
    values = mutual_information(MEMBERS)
    assert len(values) == 2
    assert all(value >= -1e-12 for value in values)


def test_variation_ratio_measures_disagreement():
    assert variation_ratio([1, 1, 1]) == pytest.approx(0.0)
    assert variation_ratio([0, 1, 0, 1]) == pytest.approx(0.5)


def test_summarize_ensemble_returns_all_uncertainty_measures():
    summaries = summarize_ensemble(MEMBERS, base=2)
    assert len(summaries) == 2
    assert summaries[0].mean_probabilities == pytest.approx([0.8, 0.2])
    assert 0 <= summaries[0].predictive_entropy <= 1
    assert summaries[0].mutual_information >= 0
    assert 0 <= summaries[0].variation_ratio <= 1


@pytest.mark.parametrize("probabilities", [[], [[0.5]], [[0.5, 0.6]], [[-0.1, 1.1]], "abc"])
def test_probability_rows_validate_shape_and_normalization(probabilities):
    with pytest.raises((TypeError, ValueError)):
        ensemble_mean([probabilities])


def test_ensemble_mean_rejects_inconsistent_member_shapes():
    with pytest.raises(ValueError, match="same shape"):
        ensemble_mean([[[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]])


def test_regression_ensemble_mean_and_variance_are_per_sample():
    predictions = [[1.0, 3.0], [3.0, 5.0], [2.0, 4.0]]
    assert ensemble_regression_mean(predictions) == pytest.approx([2.0, 4.0])
    assert ensemble_regression_variance(predictions) == pytest.approx([2 / 3, 2 / 3])
    assert ensemble_regression_variance(predictions, unbiased=True) == pytest.approx([1.0, 1.0])


def test_regression_ensemble_uncertainty_validates_shape_and_values():
    with pytest.raises(ValueError, match="same length"):
        ensemble_regression_mean([[1.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="finite"):
        ensemble_regression_variance([[float("inf")]])
    with pytest.raises(ValueError, match="at least two"):
        ensemble_regression_variance([[1.0]], unbiased=True)
    with pytest.raises(TypeError, match="boolean"):
        ensemble_regression_variance([[1.0]], unbiased=1)


def test_entropy_validates_base_and_inputs():
    with pytest.raises(ValueError):
        entropy([0.5, 0.5], base=1)
    with pytest.raises(ValueError):
        entropy([0.4, 0.4])
    with pytest.raises(TypeError):
        entropy([0.5, 0.5], base=True)


def test_variation_ratio_validates_classes():
    with pytest.raises(ValueError):
        variation_ratio([])
    with pytest.raises(TypeError):
        variation_ratio([True])


def test_risk_coverage_curve_keeps_confident_correct_predictions_first():
    from deeplearn_utils import risk_coverage_curve

    points = risk_coverage_curve([0.6, 0.9, 0.8], [False, True, True])
    expected = [(1 / 3, 0.0), (2 / 3, 0.0), (1.0, 1 / 3)]
    for actual, reference in zip(points, expected):
        assert actual == pytest.approx(reference)


def test_risk_coverage_curve_validates_inputs():
    from deeplearn_utils import risk_coverage_curve

    with pytest.raises(ValueError, match="same non-zero"):
        risk_coverage_curve([], [])
    with pytest.raises(ValueError, match="between"):
        risk_coverage_curve([1.2], [True])
    with pytest.raises(TypeError, match="booleans"):
        risk_coverage_curve([0.5], [1])
