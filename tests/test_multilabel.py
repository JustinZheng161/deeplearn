import pytest

from deeplearn_utils import hamming_loss, jaccard_score, micro_f1, multilabel_macro_f1, subset_accuracy, threshold_predictions


PREDICTIONS = [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
TARGETS = [[1, 1, 1], [0, 1, 0], [1, 0, 0]]


def test_threshold_predictions_converts_probabilities():
    assert threshold_predictions([[0.2, 0.5], [0.9, 0.1]]) == [[0, 1], [1, 0]]
    assert threshold_predictions([[0.2, 0.5]], threshold=0.6) == [[0, 0]]


def test_hamming_and_subset_accuracy_measure_different_errors():
    assert hamming_loss(PREDICTIONS, TARGETS) == pytest.approx(2 / 9)
    assert subset_accuracy(PREDICTIONS, TARGETS) == pytest.approx(1 / 3)


def test_jaccard_supports_sample_and_label_averages():
    assert jaccard_score(PREDICTIONS, TARGETS) == pytest.approx((2 / 3 + 1.0 + 0.5) / 3)
    assert jaccard_score(PREDICTIONS, TARGETS, average="labels") == pytest.approx((1.0 + 1 / 3 + 1.0) / 3)


def test_multilabel_macro_f1_weights_labels_equally():
    assert multilabel_macro_f1(PREDICTIONS, TARGETS) == pytest.approx((1.0 + 0.5 + 1.0) / 3)
    assert multilabel_macro_f1(TARGETS, TARGETS) == pytest.approx(1.0)


def test_micro_f1_pools_all_label_decisions():
    assert micro_f1(PREDICTIONS, TARGETS) == pytest.approx(0.8)
    assert micro_f1(TARGETS, TARGETS) == pytest.approx(1.0)


def test_multilabel_metrics_validate_shape_and_binary_values():
    with pytest.raises(ValueError, match="same shape"):
        hamming_loss([[1]], [[1, 0]])
    with pytest.raises(ValueError):
        subset_accuracy([[2]], [[1]])
    with pytest.raises(ValueError):
        jaccard_score([[1]], [[1]], average="macro")
    with pytest.raises(ValueError, match="same shape"):
        multilabel_macro_f1([[1]], [[1, 0]])


def test_threshold_predictions_validates_probability_range():
    with pytest.raises(ValueError):
        threshold_predictions([[1.2]])
    with pytest.raises(ValueError):
        threshold_predictions([[0.5]], threshold=1.1)
