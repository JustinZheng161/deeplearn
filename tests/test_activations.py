import math

import pytest

from deeplearn_utils import (
    gelu,
    mish,
    mish_derivative,
    leaky_relu,
    leaky_relu_derivative,
    relu,
    relu_derivative,
    silu,
    silu_derivative,
    softplus,
    vector_sigmoid,
)


def test_relu_and_leaky_relu_values_and_derivatives():
    assert relu([-2, 0, 3]) == [0.0, 0.0, 3.0]
    assert relu_derivative([-2, 0, 3]) == [0.0, 0.0, 1.0]
    assert leaky_relu([-2, 0, 3], negative_slope=0.1) == pytest.approx([-0.2, 0, 3])
    assert leaky_relu_derivative([-2, 0, 3], negative_slope=0.1) == [0.1, 1.0, 1.0]


def test_sigmoid_is_stable_at_large_magnitudes():
    values = vector_sigmoid([-1000, 0, 1000])
    assert values == pytest.approx([0.0, 0.5, 1.0])
    assert all(0 <= value <= 1 for value in values)


def test_silu_and_derivative_are_consistent():
    values = [-2, 0, 2]
    assert silu(values)[1] == pytest.approx(0.0)
    assert silu_derivative(values)[1] == pytest.approx(0.5)
    for value, derivative in zip(values, silu_derivative(values)):
        step = 1e-5
        numerical = (silu([value + step])[0] - silu([value - step])[0]) / (2 * step)
        assert derivative == pytest.approx(numerical, abs=1e-5)


def test_gelu_exact_and_approximate_are_close():
    exact = gelu([-2, 0, 2], approximate=False)
    approximate = gelu([-2, 0, 2], approximate=True)
    assert exact[1] == approximate[1] == pytest.approx(0.0)
    assert approximate == pytest.approx(exact, abs=5e-4)


def test_mish_is_stable_and_supports_beta():
    values = mish([-1000, 0, 1000])
    assert values[0] == pytest.approx(0.0, abs=1e-10)
    assert values[1] == pytest.approx(0.0)
    assert values[2] == pytest.approx(1000.0)
    assert mish([2], beta=2)[0] == pytest.approx(2 * math.tanh(math.log1p(math.exp(4)) / 2))


def test_mish_derivative_matches_finite_difference():
    for value in [-2.0, 0.0, 2.0]:
        step = 1e-5
        numerical = (mish([value + step])[0] - mish([value - step])[0]) / (2 * step)
        assert mish_derivative([value])[0] == pytest.approx(numerical, abs=1e-5)


def test_softplus_is_stable_and_supports_beta():
    values = softplus([-1000, 0, 1000])
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(math.log(2))
    assert values[2] == pytest.approx(1000)
    assert softplus([2], beta=2)[0] == pytest.approx(math.log1p(math.exp(4)) / 2)


@pytest.mark.parametrize("values", [[], "abc", [True], [float("inf")]])
def test_activations_validate_input(values):
    with pytest.raises((TypeError, ValueError)):
        relu(values)


def test_activation_parameters_are_validated():
    with pytest.raises(ValueError):
        leaky_relu([1], negative_slope=-1)
    with pytest.raises(TypeError):
        gelu([1], approximate=1)
    with pytest.raises(ValueError):
        softplus([1], beta=0)
    with pytest.raises(ValueError):
        softplus([1], threshold=float("nan"))
    with pytest.raises(ValueError):
        mish([1], beta=0)
    with pytest.raises(ValueError):
        mish_derivative([1], threshold=float("inf"))
