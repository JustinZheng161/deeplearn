import pytest

from deeplearn_utils import causal_mask, pad_sequences, sliding_windows, temporal_split


def test_sliding_windows_create_inputs_and_future_horizon():
    result = sliding_windows([0, 1, 2, 3, 4, 5], 3, stride=2, horizon=2)
    assert result == [([0, 1, 2], [3, 4])]


def test_temporal_split_preserves_order_and_has_no_overlap():
    train, validation, test = temporal_split(list(range(10)), 0.2, test_fraction=0.2)
    assert train == list(range(6))
    assert validation == [6, 7]
    assert test == [8, 9]


def test_pad_sequences_supports_pre_and_post_padding_and_clipping():
    post, lengths = pad_sequences([[1, 2], [3]], max_length=3, padding="post", value=-1)
    pre, _ = pad_sequences([[1, 2, 3, 4], [5]], max_length=3, padding="pre", value=0)
    assert post == [[1, 2, -1], [3, -1, -1]]
    assert pre == [[2, 3, 4], [0, 0, 5]]
    assert lengths == [2, 1]


def test_causal_mask_controls_current_token_visibility():
    assert causal_mask(3) == [[True, False, False], [True, True, False], [True, True, True]]
    assert causal_mask(3, include_current=False) == [[False, False, False], [True, False, False], [True, True, False]]


def test_sliding_windows_reject_short_sequences():
    with pytest.raises(ValueError):
        sliding_windows([1, 2], 3)


@pytest.mark.parametrize("fraction", [-0.1, 1.0, True, "0.2"])
def test_temporal_split_validates_fractions(fraction):
    with pytest.raises((TypeError, ValueError)):
        temporal_split([1, 2, 3], validation_fraction=fraction)


def test_temporal_split_requires_training_segment():
    with pytest.raises(ValueError):
        temporal_split([1, 2], 0.5, test_fraction=0.5)


@pytest.mark.parametrize("padding", ["middle", "", 1])
def test_pad_sequences_validates_padding(padding):
    with pytest.raises(ValueError):
        pad_sequences([[1]], padding=padding)


def test_sequence_helpers_validate_integer_parameters():
    with pytest.raises(ValueError):
        causal_mask(0)
    with pytest.raises(TypeError):
        sliding_windows([1, 2], True)
    with pytest.raises(ValueError):
        pad_sequences([[1]], max_length=0)


def test_lengths_to_padding_mask_supports_left_and_right_padding():
    from deeplearn_utils import lengths_to_padding_mask

    assert lengths_to_padding_mask([2, 1], max_length=3) == [[False, False, True], [False, True, True]]
    assert lengths_to_padding_mask([2, 1], max_length=3, left_padding=True) == [[True, False, False], [True, True, False]]


@pytest.mark.parametrize("lengths", [[], [-1], [1.5], [True]])
def test_lengths_to_padding_mask_validates_lengths(lengths):
    from deeplearn_utils import lengths_to_padding_mask

    with pytest.raises((TypeError, ValueError)):
        lengths_to_padding_mask(lengths)


def test_lengths_to_padding_mask_rejects_short_max_length():
    from deeplearn_utils import lengths_to_padding_mask

    with pytest.raises(ValueError):
        lengths_to_padding_mask([1, 3], max_length=2)
