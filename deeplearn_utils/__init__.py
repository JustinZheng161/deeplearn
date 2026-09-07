"""Small, dependency-light helpers for deep-learning projects."""

from .reproducibility import (
    SeedReport,
    derive_seed,
    make_generator,
    make_worker_init_fn,
    seed_everything,
)
from .data import stratified_split
from .metrics import (
    accuracy,
    balanced_accuracy,
    expected_calibration_error,
    confusion_matrix,
    macro_f1,
    precision_recall_f1,
    top_k_accuracy,
)
from .training import (
    EarlyStopping,
    EarlyStoppingState,
    RunningAverage,
    batch_indices,
    iter_minibatches,
)

__all__ = [
    "SeedReport",
    "derive_seed",
    "make_generator",
    "make_worker_init_fn",
    "seed_everything",
    "accuracy",
    "stratified_split",
    "balanced_accuracy",
    "expected_calibration_error",
    "confusion_matrix",
    "macro_f1",
    "precision_recall_f1",
    "top_k_accuracy",
    "EarlyStopping",
    "EarlyStoppingState",
    "RunningAverage",
    "batch_indices",
    "iter_minibatches",
]
