import pytest

from deeplearn_utils import (
    distillation_kl,
    distillation_loss,
    hard_cross_entropy,
    soft_targets,
    teacher_student_agreement,
)


def test_soft_targets_are_normalized_and_temperature_softens_distribution():
    sharp = soft_targets([[4, 0]], temperature=1)[0]
    soft = soft_targets([[4, 0]], temperature=2)[0]
    assert sum(sharp) == pytest.approx(1.0)
    assert soft[0] < sharp[0]


def test_distillation_kl_is_lower_for_matching_student_logits():
    teacher = [[4, 1], [1, 4]]
    matching = distillation_kl(teacher, teacher, temperature=2)
    opposing = distillation_kl([[1, 4], [4, 1]], teacher, temperature=2)
    assert matching == pytest.approx(0.0)
    assert opposing > matching


def test_hard_cross_entropy_rewards_correct_class():
    good = hard_cross_entropy([[5, 0], [0, 5]], [0, 1])
    bad = hard_cross_entropy([[0, 5], [5, 0]], [0, 1])
    assert good < bad


def test_distillation_loss_blends_soft_and_hard_objectives():
    student = [[3, 1], [1, 3]]
    teacher = [[4, 0], [0, 4]]
    result = distillation_loss(student, teacher, [0, 1], temperature=2, alpha=0.5)
    assert result > 0
    assert result < hard_cross_entropy(student, [1, 0])


def test_teacher_student_agreement_reports_top_class_matches():
    assert teacher_student_agreement([[2, 1], [1, 2]], [[3, 0], [0, 3]]) == pytest.approx(1.0)
    assert teacher_student_agreement([[1, 2]], [[3, 0]]) == pytest.approx(0.0)


def test_distillation_validates_shapes_and_labels():
    with pytest.raises(ValueError, match="same shape"):
        distillation_kl([[1, 2]], [[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        hard_cross_entropy([[1, 2]], [2])
    with pytest.raises(ValueError):
        distillation_loss([[1, 2]], [[1, 2]], [0], alpha=1.1)


@pytest.mark.parametrize("temperature", [0, -1, True, "2"])
def test_distillation_validates_temperature(temperature):
    with pytest.raises((TypeError, ValueError)):
        soft_targets([[1, 2]], temperature=temperature)


def test_distillation_validates_empty_and_nonfinite_logits():
    with pytest.raises(ValueError):
        soft_targets([])
    with pytest.raises(ValueError):
        soft_targets([[1, float("inf")]])
