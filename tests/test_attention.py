import pytest

from deeplearn_utils import attention_pool, causal_attention_mask, scaled_dot_product_attention


def test_attention_returns_normalized_weights_and_weighted_values():
    output, weights = scaled_dot_product_attention([[1, 0]], [[1, 0], [0, 1]], [[2, 0], [0, 4]], scale=1)
    assert sum(weights[0]) == pytest.approx(1.0)
    assert output[0] == pytest.approx([1.462117, 1.075766], abs=1e-5)


def test_attention_mask_blocks_keys_and_handles_fully_masked_rows():
    output, weights = scaled_dot_product_attention(
        [[1, 0], [1, 0]], [[1, 0], [0, 1]], [[2, 0], [0, 4]], mask=[[True, False], [False, False]], scale=1
    )
    assert weights[0] == pytest.approx([1.0, 0.0])
    assert output[0] == pytest.approx([2.0, 0.0])
    assert weights[1] == [0.0, 0.0]
    assert output[1] == [0.0, 0.0]


def test_causal_mask_supports_equal_and_cross_attention_lengths():
    assert causal_attention_mask(3) == [[True, False, False], [True, True, False], [True, True, True]]
    assert causal_attention_mask(2, 4) == [[True, True, True, False], [True, True, True, True]]


def test_attention_pool_uses_first_query_output():
    pooled = attention_pool([[10, 0]], [[1, 0], [0, 1]], [[2, 0], [0, 4]])
    assert sum(pooled) == pytest.approx(2.0, abs=2e-3)


def test_attention_is_stable_for_large_scores():
    output, weights = scaled_dot_product_attention([[1000, 0]], [[1000, 0], [999, 0]], [[1, 2], [3, 4]])
    assert sum(weights[0]) == pytest.approx(1.0)
    assert all(value >= 0 for value in output[0])


@pytest.mark.parametrize("values", [[], [[1, 2], [3]], "abc"])
def test_attention_validates_matrices(values):
    with pytest.raises((TypeError, ValueError)):
        scaled_dot_product_attention(values, [[1, 2]], [[1, 2]])


def test_attention_validates_dimensions_and_mask():
    with pytest.raises(ValueError, match="same feature"):
        scaled_dot_product_attention([[1]], [[1, 2]], [[1, 2]])
    with pytest.raises(ValueError, match="mask"):
        scaled_dot_product_attention([[1, 0]], [[1, 0]], [[1, 0]], mask=[[True, False]])
    with pytest.raises(TypeError):
        scaled_dot_product_attention([[1, 0]], [[1, 0]], [[1, 0]], mask=[[1]])


def test_causal_mask_validates_lengths_and_scale():
    with pytest.raises(ValueError):
        causal_attention_mask(0)
    with pytest.raises(ValueError):
        scaled_dot_product_attention([[1]], [[1]], [[1]], scale=0)
