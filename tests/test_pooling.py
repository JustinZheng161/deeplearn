import math

import pytest

from deeplearn_utils import logsumexp_pool, masked_softmax, max_pool, mean_pool, min_pool, sum_pool


VALUES = [[1, 10], [3, 5], [2, 8]]


def test_basic_pooling_reduces_features_across_rows():
    assert mean_pool(VALUES) == pytest.approx([2.0, 23 / 3])
    assert sum_pool(VALUES) == pytest.approx([6.0, 23.0])
    assert max_pool(VALUES) == [3.0, 10.0]
    assert min_pool(VALUES) == [1.0, 5.0]


def test_masked_pooling_excludes_padding_rows():
    mask = [True, False, True]
    assert mean_pool(VALUES, mask=mask) == pytest.approx([1.5, 9.0])
    assert sum_pool(VALUES, mask=mask) == pytest.approx([3.0, 18.0])
    assert max_pool(VALUES, mask=mask) == [2.0, 10.0]
    assert min_pool(VALUES, mask=mask) == [1.0, 8.0]


def test_logsumexp_pool_is_stable_for_large_values():
    result = logsumexp_pool([[1000], [1001]])
    assert result[0] == pytest.approx(1001 + math.log1p(math.exp(-1)))


def test_masked_softmax_normalizes_and_zeroes_padding():
    result = masked_softmax([1.0, 2.0, 3.0], mask=[True, False, True])
    assert result[1] == 0.0
    assert sum(result) == pytest.approx(1.0)
    assert result[2] > result[0]


def test_masked_softmax_is_stable_for_large_values():
    result = masked_softmax([1000, 1001])
    assert result == pytest.approx([1 / (1 + math.e), math.e / (1 + math.e)])


@pytest.mark.parametrize("pool", [mean_pool, sum_pool, max_pool, min_pool, logsumexp_pool])
def test_pooling_validates_matrix_and_mask(pool):
    with pytest.raises(ValueError):
        pool([])
    with pytest.raises(ValueError):
        pool([[1, 2], [3]])
    with pytest.raises(ValueError):
        pool([[1]], mask=[])
    with pytest.raises(TypeError):
        pool([[1]], mask=[1])


def test_masked_softmax_validates_values_and_mask():
    with pytest.raises(ValueError):
        masked_softmax([])
    with pytest.raises(ValueError):
        masked_softmax([1, float("inf")])
    with pytest.raises(ValueError):
        masked_softmax([1, 2], mask=[False, False])
