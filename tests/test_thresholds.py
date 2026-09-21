import pytest

from deeplearn_utils import (
    ThresholdResult,
    best_threshold,
    binary_predictions,
    confusion_at_threshold,
)


PROBABILITIES = [0.05, 0.2, 0.4, 0.65, 0.8, 0.95]
LABELS = [0, 0, 1, 1, 1, 1]


def test_confusion_at_threshold_returns_counts():
    assert confusion_at_threshold(PROBABILITIES, LABELS, threshold=0.5) == (3, 0, 2, 1)
    assert confusion_at_threshold(PROBABILITIES, LABELS, threshold=0.0) == (4, 2, 0, 0)


def test_binary_predictions_uses_inclusive_threshold():
    assert binary_predictions([0.1, 0.5, 0.9], threshold=0.5) == [0, 1, 1]


def test_best_threshold_maximizes_f1_and_returns_confusion():
    result = best_threshold(PROBABILITIES, LABELS, objective="f1")
    assert isinstance(result, ThresholdResult)
    assert result.objective == "f1"
    assert result.threshold == pytest.approx(0.4)
    assert result.score == pytest.approx(1.0)
    assert (result.true_positive, result.false_positive, result.true_negative, result.false_negative) == (4, 0, 2, 0)


def test_best_threshold_supports_balanced_accuracy_and_youden():
    balanced = best_threshold(PROBABILITIES, LABELS, objective="balanced_accuracy")
    youden = best_threshold(PROBABILITIES, LABELS, objective="youden")
    assert balanced.score == pytest.approx(1.0)
    assert youden.score == pytest.approx(1.0)
    assert balanced.threshold == youden.threshold


def test_best_threshold_tie_prefers_half_when_possible():
    result = best_threshold([0.2, 0.8], [0, 1], objective="f1")
    assert result.threshold == pytest.approx(0.5)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, True, "0.5"])
def test_threshold_validates_threshold_value(threshold):
    with pytest.raises((TypeError, ValueError)):
        binary_predictions([0.5], threshold)


@pytest.mark.parametrize("objective", ["accuracy", "loss", ""])
def test_best_threshold_validates_objective(objective):
    with pytest.raises(ValueError, match="objective"):
        best_threshold([0.5], [1], objective=objective)


@pytest.mark.parametrize("probabilities, labels", [([], []), ([0.5], []), ([1.2], [1]), ([0.5], [2])])
def test_threshold_helpers_validate_inputs(probabilities, labels):
    with pytest.raises((TypeError, ValueError)):
        confusion_at_threshold(probabilities, labels)


def test_binary_predictions_does_not_mutate_input():
    values = [0.2, 0.8]
    binary_predictions(values)
    assert values == [0.2, 0.8]
