import pytest

from deeplearn_utils import average_precision, best_f1_threshold, precision_recall_curve, roc_auc


SCORES = [0.9, 0.8, 0.4, 0.1]
LABELS = [1, 0, 1, 0]


def test_roc_auc_handles_ordering_and_ties():
    assert roc_auc([0.9, 0.8, 0.4, 0.1], [1, 0, 1, 0]) == pytest.approx(0.75)
    assert roc_auc([0.5, 0.5, 0.1, 0.1], [1, 0, 0, 1]) == pytest.approx(0.5)


def test_precision_recall_curve_returns_one_point_per_unique_score():
    precision, recall, thresholds = precision_recall_curve(SCORES, LABELS)
    assert len(precision) == len(recall) == len(thresholds) == 4
    assert recall[-1] == pytest.approx(1.0)
    assert all(0 <= value <= 1 for value in precision + recall)


def test_average_precision_rewards_better_ranking():
    good = average_precision([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    poor = average_precision([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0])
    assert good > poor
    assert good == pytest.approx(1.0)


def test_best_f1_threshold_returns_optimal_operating_point():
    threshold, score = best_f1_threshold([0.9, 0.8, 0.3, 0.1], [1, 1, 0, 0])
    assert threshold == pytest.approx(0.8)
    assert score == pytest.approx(1.0)


def test_ranking_metrics_validate_binary_labels_and_class_balance():
    with pytest.raises(ValueError):
        roc_auc([0.1, 0.2], [1, 1])
    with pytest.raises(ValueError):
        average_precision([0.1], [2])
    with pytest.raises(ValueError):
        best_f1_threshold([], [])


def test_scores_must_be_finite_and_lengths_must_match():
    with pytest.raises(ValueError):
        roc_auc([0.1], [0, 1])
    with pytest.raises(ValueError):
        roc_auc([float("nan"), 0.2], [1, 0])
