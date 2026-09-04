"""Small, dependency-light helpers for deep-learning projects."""

from .reproducibility import (
    SeedReport,
    derive_seed,
    make_generator,
    make_worker_init_fn,
    seed_everything,
)
from .metrics import (
    accuracy,
    balanced_accuracy,
    confusion_matrix,
    macro_f1,
    precision_recall_f1,
    top_k_accuracy,
)

__all__ = [
    "SeedReport",
    "derive_seed",
    "make_generator",
    "make_worker_init_fn",
    "seed_everything",
    "accuracy",
    "balanced_accuracy",
    "confusion_matrix",
    "macro_f1",
    "precision_recall_f1",
    "top_k_accuracy",
]
