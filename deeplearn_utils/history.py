"""Framework-agnostic history tracking for deep-learning experiments."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


Number = int | float


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True)
class EpochRecord:
    """Immutable metrics captured after one completed epoch."""

    epoch: int
    metrics: dict[str, float]

    def __post_init__(self) -> None:
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int):
            raise TypeError("epoch must be an integer")
        if self.epoch < 0:
            raise ValueError("epoch must be non-negative")
        if not self.metrics:
            raise ValueError("metrics must not be empty")
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise TypeError("metric names must be non-empty strings")
            _number(value, f"metric {name!r}")

    def value(self, name: str) -> float:
        """Return a metric from this epoch or raise a clear error."""
        try:
            return self.metrics[name]
        except KeyError as exc:
            raise KeyError(f"metric {name!r} is not present in epoch {self.epoch}") from exc


@dataclass(frozen=True)
class BestMetric:
    """The best observed value and its epoch."""

    name: str
    value: float
    epoch: int
    mode: str


class MetricHistory:
    """Collect epoch metrics and query best values without dependencies.

    Epochs must be recorded in increasing order. Missing metrics are allowed,
    which supports phases where a validation pass is not run every epoch.
    Records and best values can be exported as JSON for experiment artifacts.
    """

    def __init__(self) -> None:
        self._records: list[EpochRecord] = []

    @property
    def records(self) -> tuple[EpochRecord, ...]:
        """Return an immutable view of recorded epochs."""
        return tuple(self._records)

    @property
    def epochs(self) -> list[int]:
        """Return recorded epoch numbers in insertion order."""
        return [record.epoch for record in self._records]

    def add(self, epoch: int, metrics: Mapping[str, Number]) -> EpochRecord:
        """Record one epoch and return the immutable record.

        Metric values are copied and converted to floats so later mutation of a
        caller-owned dictionary cannot change the history.
        """
        if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        if self._records and epoch <= self._records[-1].epoch:
            raise ValueError("epoch must be greater than the previous epoch")
        normalized = {name: _number(value, f"metric {name!r}") for name, value in metrics.items()}
        record = EpochRecord(epoch, normalized)
        self._records.append(record)
        return record

    def values(self, name: str) -> list[float | None]:
        """Return a metric series, using ``None`` for missing epochs."""
        if not isinstance(name, str) or not name:
            raise TypeError("name must be a non-empty string")
        return [record.metrics.get(name) for record in self._records]

    def best(self, name: str, *, mode: str = "min") -> BestMetric:
        """Return the best observed value for ``name`` and its epoch."""
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        candidates = [record for record in self._records if name in record.metrics]
        if not candidates:
            raise KeyError(f"metric {name!r} has no recorded values")
        key = lambda record: record.metrics[name]
        winner = min(candidates, key=key) if mode == "min" else max(candidates, key=key)
        return BestMetric(name, winner.metrics[name], winner.epoch, mode)

    def latest(self, name: str) -> float:
        """Return the most recent value for a metric."""
        for record in reversed(self._records):
            if name in record.metrics:
                return record.metrics[name]
        raise KeyError(f"metric {name!r} has no recorded values")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible history representation."""
        return {"records": [asdict(record) for record in self._records]}

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the history to JSON with deterministic key ordering."""
        if indent is not None and (isinstance(indent, bool) or not isinstance(indent, int)):
            raise TypeError("indent must be an integer or None")
        if indent is not None and indent < 0:
            raise ValueError("indent must be non-negative")
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricHistory":
        """Restore history from :meth:`to_dict` output."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        records = payload.get("records")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ValueError("payload records must be a sequence")
        history = cls()
        for item in records:
            if not isinstance(item, Mapping):
                raise ValueError("each record must be a mapping")
            if "epoch" not in item or "metrics" not in item:
                raise ValueError("each record needs epoch and metrics")
            history.add(item["epoch"], item["metrics"])
        return history

    @classmethod
    def from_json(cls, payload: str) -> "MetricHistory":
        """Restore history from a JSON string."""
        if not isinstance(payload, str):
            raise TypeError("payload must be a string")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("payload is not valid JSON") from exc
        return cls.from_dict(decoded)

    def clear(self) -> None:
        """Remove all records so the object can be reused."""
        self._records.clear()


__all__ = ["BestMetric", "EpochRecord", "MetricHistory"]
