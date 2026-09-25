"""Small, dependency-light helpers for deep-learning projects."""

from .reproducibility import (
    SeedReport,
    derive_seed,
    make_generator,
    make_worker_init_fn,
    seed_everything,
)
from .data import stratified_kfold, stratified_split
from .schedules import warmup_cosine_decay
from .regression import (
    huber_loss,
    huber_loss_batch,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from .targets import label_smoothed_targets
from .preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from .probabilities import (
    brier_score,
    brier_score_batch,
    categorical_cross_entropy,
    categorical_cross_entropy_batch,
)
from .retrieval import ndcg_at_k, precision_at_k
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
from .history import BestMetric, EpochRecord, MetricHistory
from .losses import binary_focal_loss, dice_coefficient, dice_loss, multiclass_focal_loss
from .augmentation import (
    add_gaussian_noise,
    compose_augmentations,
    mixup,
    random_feature_dropout,
    inverted_feature_dropout,
)
from .validation import fold_class_counts, kfold_indices, stratified_kfold_indices
from .probabilities import (
    binary_cross_entropy,
    categorical_entropy,
    log_softmax,
    sigmoid,
    softmax,
    softmax_batch,
)
from .thresholds import ThresholdResult, best_threshold, binary_predictions, confusion_at_threshold
from .retrieval import (
    cosine_similarity,
    mean_average_precision,
    mean_reciprocal_rank,
    ranked_indices,
    recall_at_k,
    similarity_matrix,
)

__all__ = [
    "SeedReport",
    "derive_seed",
    "make_generator",
    "make_worker_init_fn",
    "seed_everything",
    "accuracy",
    "ndcg_at_k",
    "precision_at_k",
    "brier_score",
    "brier_score_batch",
    "categorical_cross_entropy",
    "categorical_cross_entropy_batch",
    "stratified_kfold",
    "stratified_split",
    "warmup_cosine_decay",
    "huber_loss",
    "huber_loss_batch",
    "mean_absolute_error",
    "mean_squared_error",
    "r2_score",
    "label_smoothed_targets",
    "MinMaxScaler",
    "RobustScaler",
    "StandardScaler",
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
    "BestMetric",
    "EpochRecord",
    "MetricHistory",
    "MinMaxScaler",
    "StandardScaler",
    "add_gaussian_noise",
    "compose_augmentations",
    "mixup",
    "random_feature_dropout",
    "inverted_feature_dropout",
    "fold_class_counts",
    "kfold_indices",
    "stratified_kfold_indices",
    "binary_cross_entropy",
    "categorical_entropy",
    "log_softmax",
    "sigmoid",
    "softmax",
    "softmax_batch",
    "ThresholdResult",
    "best_threshold",
    "binary_predictions",
    "confusion_at_threshold",
    "cosine_similarity",
    "mean_average_precision",
    "mean_reciprocal_rank",
    "ranked_indices",
    "recall_at_k",
    "similarity_matrix",
    "binary_focal_loss",
    "dice_coefficient",
    "dice_loss",
    "multiclass_focal_loss",
]
