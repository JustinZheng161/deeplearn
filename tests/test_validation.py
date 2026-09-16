import pytest

from deeplearn_utils import fold_class_counts, kfold_indices, stratified_kfold_indices


def test_kfold_covers_each_sample_once_in_validation():
    folds = list(kfold_indices(10, folds=3))
    validation = [index for _, current in folds for index in current]
    assert sorted(validation) == list(range(10))
    assert [len(current) for _, current in folds] == [4, 3, 3]
    for train, current in folds:
        assert set(train).isdisjoint(current)
        assert sorted(train + current) == list(range(10))


def test_kfold_shuffle_is_reproducible_and_local():
    first = list(kfold_indices(8, folds=4, shuffle=True, seed=5))
    assert first == list(kfold_indices(8, folds=4, shuffle=True, seed=5))
    assert first != list(kfold_indices(8, folds=4, shuffle=True, seed=6))


def test_stratified_kfold_distributes_each_class_across_folds():
    labels = [0] * 8 + [1] * 5 + [2] * 2
    folds = list(stratified_kfold_indices(labels, folds=3, shuffle=True, seed=9))
    validation = [index for _, current in folds for index in current]
    assert sorted(validation) == list(range(len(labels)))
    counts = [fold_class_counts(labels, current) for _, current in folds]
    assert [item.get(0, 0) for item in counts] == [3, 3, 2]
    assert sorted(item.get(1, 0) for item in counts) == [1, 2, 2]
    assert sorted(item.get(2, 0) for item in counts) == [0, 1, 1]


def test_stratified_kfold_keeps_singletons_out_of_most_validation_folds():
    folds = list(stratified_kfold_indices([0, 0, 1, 2], folds=3, seed=1))
    singleton_occurrences = [2 in validation for _, validation in folds]
    assert sum(singleton_occurrences) == 1


def test_stratified_kfold_preserves_training_complement():
    labels = ["cat", "cat", "dog", "dog", "bird", "bird"]
    for train, validation in stratified_kfold_indices(labels, folds=3, seed=4):
        assert set(train).isdisjoint(validation)
        assert sorted(train + validation) == list(range(len(labels)))


def test_fold_class_counts_counts_and_validates_indices():
    assert fold_class_counts(["a", "a", "b"], [0, 2, 2]) == {"a": 1, "b": 2}
    with pytest.raises(ValueError):
        fold_class_counts([0], [1])
    with pytest.raises(TypeError):
        fold_class_counts([0], [True])


@pytest.mark.parametrize(
    ("size", "folds", "error"),
    [(True, 2, TypeError), (-1, 2, ValueError), (4, 1, ValueError), (2, 3, ValueError)],
)
def test_kfold_validates_parameters(size, folds, error):
    with pytest.raises(error):
        list(kfold_indices(size, folds))


@pytest.mark.parametrize("labels", [[], "abc", [[0], [1]]])
def test_stratified_kfold_validates_labels(labels):
    with pytest.raises((TypeError, ValueError)):
        list(stratified_kfold_indices(labels, folds=2))


def test_stratified_kfold_validates_seed_and_fold_count():
    with pytest.raises(TypeError):
        list(stratified_kfold_indices([0, 1], folds=2, seed=True))
    with pytest.raises(ValueError):
        list(stratified_kfold_indices([0, 1], folds=1))
