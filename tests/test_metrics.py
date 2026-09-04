import pytest

from deeplearn_utils import (
    accuracy,
    confusion_matrix,
    macro_f1,
    precision_recall_f1,
    top_k_accuracy,
)


def test_accuracy_counts_exact_matches():
    assert accuracy([0, 1, 2, 1], [0, 2, 2, 1]) == pytest.approx(0.75)


def test_accuracy_rejects_empty_or_mismatched_inputs():
    with pytest.raises(ValueError, match="must not be empty"):
        accuracy([], [])
    with pytest.raises(ValueError, match="same length"):
        accuracy([1], [1, 1])


def test_confusion_matrix_discovers_first_seen_class_order():
    assert confusion_matrix(["cat", "dog", "cat"], ["dog", "dog", "cat"]) == [
        [1, 1],
        [0, 1],
    ]


def test_confusion_matrix_respects_explicit_labels():
    assert confusion_matrix([1, 0, 1], [0, 0, 1], labels=[0, 1]) == [[1, 0], [1, 1]]


def test_confusion_matrix_rejects_unknown_and_duplicate_labels():
    with pytest.raises(ValueError, match="do not include"):
        confusion_matrix([0, 1], [0, 1], labels=[0])
    with pytest.raises(ValueError, match="duplicates"):
        confusion_matrix([0], [0], labels=[0, 0])


def test_binary_precision_recall_f1_returns_support():
    result = precision_recall_f1([1, 1, 0, 0], [1, 0, 1, 0])
    assert result == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2.0}


def test_binary_metrics_are_zero_when_positive_class_is_absent():
    assert precision_recall_f1([0, 0], [0, 0], positive_label=1) == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0.0,
    }


def test_macro_f1_averages_all_observed_classes():
    assert macro_f1([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(11 / 15)


def test_top_k_accuracy_accepts_ties_deterministically():
    targets = [0, 1, 2]
    scores = [[0.5, 0.5, 0.1], [0.2, 0.2, 0.1], [0.1, 0.2, 0.3]]
    assert top_k_accuracy(targets, scores, k=1) == pytest.approx(2 / 3)
    assert top_k_accuracy(targets, scores, k=2) == 1.0


@pytest.mark.parametrize(
    ("k", "error"), [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)]
)
def test_top_k_accuracy_validates_k(k, error):
    with pytest.raises(error):
        top_k_accuracy([0], [[1.0]], k=k)


@pytest.mark.parametrize(
    "scores",
    [[], [[1.0], [0.5]], [[float("nan")]], [[1.0, 0.0], [1.0]]],
)
def test_top_k_accuracy_rejects_invalid_score_rows(scores):
    with pytest.raises(ValueError):
        top_k_accuracy([0], scores)


def test_top_k_accuracy_rejects_non_integer_or_out_of_range_targets():
    with pytest.raises(TypeError):
        top_k_accuracy(["0"], [[1.0]])
    with pytest.raises(ValueError):
        top_k_accuracy([2], [[1.0, 0.0]])
