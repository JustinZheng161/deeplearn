import json

import pytest

from deeplearn_utils import BestMetric, EpochRecord, MetricHistory


def test_add_records_and_query_metric_series():
    history = MetricHistory()
    history.add(0, {"loss": 1.0, "accuracy": 0.4})
    history.add(1, {"loss": 0.6})
    history.add(2, {"loss": 0.4, "accuracy": 0.8})

    assert history.epochs == [0, 1, 2]
    assert history.values("accuracy") == [0.4, None, 0.8]
    assert history.latest("loss") == 0.4
    assert history.records[0] == EpochRecord(0, {"loss": 1.0, "accuracy": 0.4})


def test_best_metric_supports_min_and_max_modes():
    history = MetricHistory()
    history.add(0, {"loss": 1.0, "accuracy": 0.4})
    history.add(1, {"loss": 0.5, "accuracy": 0.7})
    history.add(2, {"loss": 0.7, "accuracy": 0.6})

    assert history.best("loss") == BestMetric("loss", 0.5, 1, "min")
    assert history.best("accuracy", mode="max") == BestMetric("accuracy", 0.7, 1, "max")


def test_history_round_trips_through_dict_and_json():
    history = MetricHistory()
    history.add(0, {"loss": 1})
    history.add(1, {"loss": 0.5, "lr": 0.01})

    restored = MetricHistory.from_dict(history.to_dict())
    assert restored.records == history.records
    assert MetricHistory.from_json(history.to_json()).records == history.records
    assert json.loads(history.to_json())["records"][1]["metrics"]["lr"] == 0.01


def test_history_copies_input_metrics():
    metrics = {"loss": 1.0}
    history = MetricHistory()
    history.add(0, metrics)
    metrics["loss"] = 999
    assert history.latest("loss") == 1.0


def test_history_clear_allows_reuse():
    history = MetricHistory()
    history.add(0, {"loss": 1})
    history.clear()
    assert history.records == ()
    history.add(0, {"loss": 2})
    assert history.latest("loss") == 2.0


@pytest.mark.parametrize(
    "payload",
    [{}, {"records": "bad"}, {"records": [{"epoch": 0}]}, {"records": [1]}],
)
def test_from_dict_rejects_malformed_payloads(payload):
    with pytest.raises((TypeError, ValueError)):
        MetricHistory.from_dict(payload)


@pytest.mark.parametrize(
    ("epoch", "metrics", "error"),
    [(True, {"loss": 1}, TypeError), (-1, {"loss": 1}, ValueError), (0, {}, ValueError)],
)
def test_add_validates_epoch_and_metrics(epoch, metrics, error):
    with pytest.raises(error):
        MetricHistory().add(epoch, metrics)


def test_history_requires_increasing_epochs_and_known_metrics():
    history = MetricHistory()
    history.add(2, {"loss": 1})
    with pytest.raises(ValueError, match="greater"):
        history.add(2, {"loss": 0.5})
    with pytest.raises(KeyError):
        history.best("accuracy")
    with pytest.raises(KeyError):
        history.latest("accuracy")


def test_history_validates_json_and_indent():
    with pytest.raises(ValueError, match="valid JSON"):
        MetricHistory.from_json("{")
    history = MetricHistory()
    with pytest.raises(TypeError):
        history.to_json(indent=True)
    with pytest.raises(ValueError):
        history.to_json(indent=-1)


def test_best_rejects_unknown_mode_and_values_rejects_empty_name():
    history = MetricHistory()
    history.add(0, {"loss": 1})
    with pytest.raises(ValueError):
        history.best("loss", mode="median")
    with pytest.raises(TypeError):
        history.values("")
