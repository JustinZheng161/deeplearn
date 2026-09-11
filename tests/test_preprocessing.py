import pytest

from deeplearn_utils import MinMaxScaler, StandardScaler


DATA = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]


def test_standard_scaler_centers_and_scales_features():
    scaler = StandardScaler()
    transformed = scaler.fit_transform(DATA)
    expected = [[-1.224745, -1.224745], [0.0, 0.0], [1.224745, 1.224745]]
    for actual_row, expected_row in zip(transformed, expected):
        assert actual_row == pytest.approx(expected_row, abs=1e-5)
    assert scaler.n_features_in_ == 2
    assert scaler.n_samples_seen_ == 3


def test_standard_scaler_inverse_transform_round_trips():
    scaler = StandardScaler().fit(DATA)
    transformed = scaler.transform([[1.5, 15.0]])
    assert scaler.inverse_transform(transformed)[0] == pytest.approx([1.5, 15.0])


def test_standard_scaler_handles_constant_columns():
    scaler = StandardScaler().fit([[2, 5], [2, 7]])
    assert scaler.transform([[2, 6]]) == [[0.0, 0.0]]
    assert scaler.inverse_transform([[0.0, 0.5]])[0] == pytest.approx([2.0, 6.5])


def test_minmax_scaler_supports_custom_range_and_inverse():
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    transformed = scaler.fit_transform(DATA)
    for actual_row, expected_row in zip(transformed, [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]):
        assert actual_row == pytest.approx(expected_row)
    assert scaler.inverse_transform([[0.5, -0.5]])[0] == pytest.approx([2.5, 15.0])


def test_minmax_scaler_constant_columns_map_to_lower_bound():
    scaler = MinMaxScaler((2, 4)).fit([[3, 1], [3, 5]])
    assert scaler.transform([[3, 3]]) == [[2.0, 3.0]]
    assert scaler.inverse_transform([[4, 3]])[0] == pytest.approx([3.0, 3.0])


def test_scalers_require_fit_before_transform():
    with pytest.raises(RuntimeError, match="not been fitted"):
        StandardScaler().transform([[1]])
    with pytest.raises(RuntimeError, match="not been fitted"):
        MinMaxScaler().inverse_transform([[1]])


def test_scalers_reject_feature_width_mismatch():
    scaler = StandardScaler().fit(DATA)
    with pytest.raises(ValueError, match="expected 2"):
        scaler.transform([[1, 2, 3]])
    minimum = MinMaxScaler().fit(DATA)
    with pytest.raises(ValueError, match="expected 2"):
        minimum.inverse_transform([[1]])


@pytest.mark.parametrize("data", [[], [[1, 2], [3]], "abc"])
def test_scalers_reject_empty_or_ragged_data(data):
    with pytest.raises((TypeError, ValueError)):
        StandardScaler().fit(data)


@pytest.mark.parametrize(
    "feature_range",
    [(0, 0), (1, 0), (0,), [0, 1], (float("nan"), 1)],
)
def test_minmax_scaler_validates_range(feature_range):
    with pytest.raises((TypeError, ValueError)):
        MinMaxScaler(feature_range)


def test_scalers_reject_nonfinite_and_non_numeric_values():
    with pytest.raises(ValueError):
        StandardScaler().fit([[1, float("inf")]])
    with pytest.raises(TypeError):
        MinMaxScaler().fit([[1, "x"]])


def test_fit_copies_input_and_does_not_mutate_rows():
    source = [[1, 2], [3, 4]]
    StandardScaler().fit_transform(source)
    assert source == [[1, 2], [3, 4]]
