"""Leakage-safe feature preprocessing for deep-learning experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def _matrix(data: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    if isinstance(data, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of rows")
    try:
        rows = [list(row) for row in data]
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of rows") from exc
    if not rows:
        raise ValueError(f"{name} must not be empty")
    width = len(rows[0])
    if width == 0:
        raise ValueError(f"{name} rows must not be empty")
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be rectangular")
    for row in rows:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} values must be numbers")
            if not math.isfinite(value):
                raise ValueError(f"{name} values must be finite")
    return [[float(value) for value in row] for row in rows]


def _validate_fitted_width(data: list[list[float]], width: int) -> None:
    if len(data[0]) != width:
        raise ValueError(f"expected {width} features, received {len(data[0])}")


class StandardScaler:
    """Center features and scale them by population standard deviation.

    Statistics are learned only by :meth:`fit`, so callers can fit on the
    training split and transform validation data without leakage. Constant
    columns use scale one, mapping them to zero without division errors.
    """

    def __init__(self) -> None:
        self.mean_: list[float] | None = None
        self.scale_: list[float] | None = None
        self.n_samples_seen_: int = 0

    @property
    def n_features_in_(self) -> int:
        """Return the fitted feature count."""
        if self.mean_ is None:
            raise RuntimeError("scaler has not been fitted")
        return len(self.mean_)

    def fit(self, data: Sequence[Sequence[float]]) -> "StandardScaler":
        """Learn per-feature means and standard deviations."""
        rows = _matrix(data, "data")
        count = len(rows)
        width = len(rows[0])
        means = [sum(row[column] for row in rows) / count for column in range(width)]
        variances = [
            sum((row[column] - means[column]) ** 2 for row in rows) / count
            for column in range(width)
        ]
        self.mean_ = means
        self.scale_ = [math.sqrt(variance) or 1.0 for variance in variances]
        self.n_samples_seen_ = count
        return self

    def _require_fitted(self) -> tuple[list[float], list[float]]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("scaler has not been fitted")
        return self.mean_, self.scale_

    def transform(self, data: Sequence[Sequence[float]]) -> list[list[float]]:
        """Standardize rows using statistics learned during fitting."""
        means, scales = self._require_fitted()
        rows = _matrix(data, "data")
        _validate_fitted_width(rows, len(means))
        return [[(value - means[column]) / scales[column] for column, value in enumerate(row)] for row in rows]

    def inverse_transform(self, data: Sequence[Sequence[float]]) -> list[list[float]]:
        """Convert standardized rows back to their original units."""
        means, scales = self._require_fitted()
        rows = _matrix(data, "data")
        _validate_fitted_width(rows, len(means))
        return [[value * scales[column] + means[column] for column, value in enumerate(row)] for row in rows]

    def fit_transform(self, data: Sequence[Sequence[float]]) -> list[list[float]]:
        """Fit on data and return its standardized representation."""
        return self.fit(data).transform(data)


class MinMaxScaler:
    """Scale each feature into a configurable interval."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        if not isinstance(feature_range, tuple) or len(feature_range) != 2:
            raise TypeError("feature_range must be a two-item tuple")
        lower, upper = feature_range
        for name, value in (("lower", lower), ("upper", upper)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} bound must be a number")
            if not math.isfinite(value):
                raise ValueError(f"{name} bound must be finite")
        if lower >= upper:
            raise ValueError("feature_range lower bound must be less than upper bound")
        self.feature_range = (float(lower), float(upper))
        self.data_min_: list[float] | None = None
        self.data_max_: list[float] | None = None
        self.n_samples_seen_: int = 0

    @property
    def n_features_in_(self) -> int:
        """Return the fitted feature count."""
        if self.data_min_ is None:
            raise RuntimeError("scaler has not been fitted")
        return len(self.data_min_)

    def fit(self, data: Sequence[Sequence[float]]) -> "MinMaxScaler":
        """Learn per-feature extrema."""
        rows = _matrix(data, "data")
        width = len(rows[0])
        self.data_min_ = [min(row[column] for row in rows) for column in range(width)]
        self.data_max_ = [max(row[column] for row in rows) for column in range(width)]
        self.n_samples_seen_ = len(rows)
        return self

    def _require_fitted(self) -> tuple[list[float], list[float]]:
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("scaler has not been fitted")
        return self.data_min_, self.data_max_

    def transform(self, data: Sequence[Sequence[float]]) -> list[list[float]]:
        """Scale rows into ``feature_range``."""
        minima, maxima = self._require_fitted()
        rows = _matrix(data, "data")
        _validate_fitted_width(rows, len(minima))
        lower, upper = self.feature_range
        width = upper - lower
        return [
            [lower if minima[column] == maxima[column] else lower + (value - minima[column]) / (maxima[column] - minima[column]) * width for column, value in enumerate(row)]
            for row in rows
        ]

    def inverse_transform(self, data: Sequence[Sequence[float]]) -> list[list[float]]:
        """Map scaled rows back to the original feature units."""
        minima, maxima = self._require_fitted()
        rows = _matrix(data, "data")
        _validate_fitted_width(rows, len(minima))
        lower, upper = self.feature_range
        width = upper - lower
        return [
            [minima[column] if minima[column] == maxima[column] else minima[column] + (value - lower) / width * (maxima[column] - minima[column]) for column, value in enumerate(row)]
            for row in rows
        ]

    def fit_transform(self, data: Sequence[Sequence[float]]) -> list[list[float]]:
        """Fit on data and return its scaled representation."""
        return self.fit(data).transform(data)


__all__ = ["MinMaxScaler", "StandardScaler"]
