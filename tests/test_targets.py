import pytest

from deeplearn_utils import label_smoothed_targets


def test_label_smoothed_targets_distributes_smoothing_mass():
    result = label_smoothed_targets([0, 2], 3, smoothing=0.2)

    assert result[0] == pytest.approx([0.8, 0.1, 0.1])
    assert result[1] == pytest.approx([0.1, 0.1, 0.8])
    assert all(sum(row) == pytest.approx(1.0) for row in result)


def test_zero_smoothing_produces_one_hot_targets():
    assert label_smoothed_targets([1], 3, smoothing=0.0) == [[0.0, 1.0, 0.0]]


@pytest.mark.parametrize("label", [-1, 3])
def test_label_smoothed_targets_rejects_out_of_range_labels(label):
    with pytest.raises(ValueError, match="outside"):
        label_smoothed_targets([label], 3)


def test_label_smoothed_targets_validates_configuration():
    with pytest.raises(ValueError, match="at least two"):
        label_smoothed_targets([0], 1)
    with pytest.raises(ValueError, match="between"):
        label_smoothed_targets([0], 2, smoothing=1.0)
    with pytest.raises(TypeError, match="integer"):
        label_smoothed_targets([0], 2.0)
    with pytest.raises(TypeError, match="only integer"):
        label_smoothed_targets([True], 2)
