import pytest

from deeplearn_utils import RunningBatchNorm, batch_norm, layer_norm, rms_norm


def test_layer_norm_zero_centers_each_row():
    result = layer_norm([[1, 2, 3], [10, 10, 10]])
    assert sum(result[0]) == pytest.approx(0.0)
    assert result[1] == pytest.approx([0.0, 0.0, 0.0])


def test_layer_norm_applies_affine_parameters():
    result = layer_norm([[1, 2]], gamma=[2, 3], beta=[1, -1])
    assert result[0] == pytest.approx([-1.0, 2.0], abs=1e-4)


def test_rms_norm_preserves_sign_and_scales_magnitude():
    result = rms_norm([[3, 4]])
    assert result[0] == pytest.approx([0.848528, 1.131371], abs=1e-5)


def test_batch_norm_zero_centers_each_feature_column():
    result = batch_norm([[1, 10], [3, 14], [5, 18]])
    assert [row[0] for row in result] == pytest.approx([-1.2247, 0.0, 1.2247], abs=1e-4)
    assert [row[1] for row in result] == pytest.approx([-1.2247, 0.0, 1.2247], abs=1e-4)


def test_running_batch_norm_updates_and_transforms_with_running_stats():
    norm = RunningBatchNorm(features=2, momentum=1.0)
    norm.update([[1, 10], [3, 14]])
    assert norm.running_mean == pytest.approx([2, 12])
    transformed = norm.transform([[2, 12]])
    assert transformed[0] == pytest.approx([0, 0])


def test_running_batch_norm_supports_multiple_updates():
    norm = RunningBatchNorm(features=1, momentum=0.5)
    norm.update([[0], [2]])
    norm.update([[4], [6]])
    assert norm.running_mean == pytest.approx([2.75])


@pytest.mark.parametrize("normalizer", [layer_norm, rms_norm, batch_norm])
def test_normalizers_validate_rectangular_values(normalizer):
    with pytest.raises(ValueError):
        normalizer([[1, 2], [3]])
    with pytest.raises(ValueError):
        normalizer([])


def test_normalizers_validate_affine_dimensions_and_epsilon():
    with pytest.raises(ValueError):
        layer_norm([[1, 2]], gamma=[1])
    with pytest.raises(ValueError):
        batch_norm([[1, 2]], epsilon=0)
    with pytest.raises(ValueError):
        rms_norm([[1, 2]], gamma=[1, float("inf")])


def test_running_batch_norm_validates_feature_count_and_hyperparameters():
    with pytest.raises(ValueError):
        RunningBatchNorm(features=0)
    with pytest.raises(ValueError):
        RunningBatchNorm(features=1, momentum=0)
    norm = RunningBatchNorm(features=2)
    with pytest.raises(ValueError):
        norm.update([[1]])
    with pytest.raises(ValueError):
        norm.transform([[1]])


def test_group_norm_normalizes_each_feature_group_independently():
    from deeplearn_utils import group_norm

    result = group_norm([[1, 2, 10, 14]], groups=2)
    assert result[0][:2] == pytest.approx([-1.0, 1.0], abs=1e-4)
    assert result[0][2:] == pytest.approx([-1.0, 1.0], abs=1e-4)


def test_group_norm_applies_affine_parameters_and_validates_groups():
    from deeplearn_utils import group_norm

    result = group_norm([[1, 3]], groups=1, gamma=[2, 3], beta=[1, -1])
    assert result[0] == pytest.approx([-1.0, 2.0], abs=1e-4)
    with pytest.raises(ValueError):
        group_norm([[1, 2, 3]], groups=2)
    with pytest.raises(TypeError):
        group_norm([[1, 2]], groups=True)
