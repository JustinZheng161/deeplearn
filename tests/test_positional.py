import math

import pytest

from deeplearn_utils import add_position_encoding, learned_position_lookup, rotate_pair, rotate_sequence, rotary_frequencies, sinusoidal_encoding


def test_sinusoidal_encoding_has_expected_shape_and_first_row():
    encoding = sinusoidal_encoding(3, 4)
    assert len(encoding) == 3
    assert all(len(row) == 4 for row in encoding)
    assert encoding[0] == pytest.approx([0.0, 1.0, 0.0, 1.0])
    assert encoding[1][0] == pytest.approx(math.sin(1.0))


def test_sinusoidal_offset_continues_the_same_sequence():
    full = sinusoidal_encoding(5, 4)
    offset = sinusoidal_encoding(2, 4, offset=3)
    assert offset[0] == pytest.approx(full[3])
    assert offset[1] == pytest.approx(full[4])


def test_learned_position_lookup_gathers_rows():
    table = [[0, 1], [2, 3], [4, 5]]
    assert learned_position_lookup([2, 0], table) == [[4.0, 5.0], [0.0, 1.0]]


def test_add_position_encoding_combines_hidden_states():
    result = add_position_encoding([[1, 2], [3, 4]], [[0.1, 0.2], [0.3, 0.4]])
    assert result[0] == pytest.approx([1.1, 2.2])
    assert result[1] == pytest.approx([3.3, 4.4])


def test_rotary_frequencies_require_even_dimension():
    frequencies = rotary_frequencies(4)
    assert len(frequencies) == 2
    assert frequencies[0] == pytest.approx(1.0)
    assert frequencies[1] < frequencies[0]


def test_rotate_pair_preserves_vector_norm():
    original = [3.0, 4.0, 1.0, -2.0]
    rotated = rotate_pair(original, position=7)
    assert sum(value * value for value in rotated) == pytest.approx(sum(value * value for value in original))
    assert rotate_pair(original, position=0) == pytest.approx(original)


def test_rotate_sequence_applies_incrementing_positions():
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    result = rotate_sequence(vectors, offset=3)
    assert result[0] == pytest.approx(rotate_pair(vectors[0], 3))
    assert result[1] == pytest.approx(rotate_pair(vectors[1], 4))
    assert [sum(value * value for value in row) for row in result] == pytest.approx([1.0, 1.0])


def test_position_helpers_validate_shape_and_ranges():
    with pytest.raises(ValueError):
        sinusoidal_encoding(0, 4)
    with pytest.raises(ValueError):
        sinusoidal_encoding(2, 0)
    with pytest.raises(ValueError):
        learned_position_lookup([3], [[1, 2]])
    with pytest.raises(ValueError):
        add_position_encoding([[1]], [[1, 2]])


def test_rotary_helpers_validate_parameters():
    with pytest.raises(ValueError):
        rotary_frequencies(3)
    with pytest.raises(ValueError):
        rotary_frequencies(4, base=1)
    with pytest.raises(ValueError):
        rotate_pair([1, 2, 3], 1)
    with pytest.raises(ValueError):
        rotate_pair([1, 2], -1)
    with pytest.raises(ValueError):
        rotate_sequence([[1, 2], [3]], offset=0)
    with pytest.raises(ValueError):
        rotate_sequence([[1, 2, 3]], offset=0)
    with pytest.raises(ValueError):
        rotate_sequence([[1, 2]], offset=-1)
