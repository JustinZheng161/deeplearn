import pytest

from deeplearn_utils import stratified_split


def test_stratified_split_preserves_each_class_in_both_sides():
    labels = [0] * 8 + [1] * 4
    train, validation = stratified_split(labels, validation_fraction=0.25, seed=7)

    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == list(range(len(labels)))
    assert [labels[index] for index in validation].count(0) == 2
    assert [labels[index] for index in validation].count(1) == 1


def test_stratified_split_is_reproducible_and_seeded():
    labels = ["cat", "cat", "dog", "dog", "bird", "bird"]

    first = stratified_split(labels, validation_fraction=0.5, seed=11)
    second = stratified_split(labels, validation_fraction=0.5, seed=11)
    third = stratified_split(labels, validation_fraction=0.5, seed=12)

    assert first == second
    assert first != third


def test_singleton_class_stays_in_training():
    train, validation = stratified_split([0, 0, 1], validation_fraction=0.5, seed=1)

    assert 2 in train
    assert 2 not in validation


@pytest.mark.parametrize(
    "fraction", [0, 1, -0.1, 1.1, True, "0.2"]
)
def test_stratified_split_validates_fraction(fraction):
    with pytest.raises((TypeError, ValueError)):
        stratified_split([0, 1], fraction)


def test_stratified_split_rejects_empty_and_unhashable_labels():
    with pytest.raises(ValueError, match="not be empty"):
        stratified_split([])
    with pytest.raises(TypeError, match="hashable"):
        stratified_split([[0], [1]])
