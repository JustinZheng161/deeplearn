"""Small, dependency-light helpers for deep-learning projects."""

from .reproducibility import (
    SeedReport,
    derive_seed,
    make_generator,
    make_worker_init_fn,
    seed_everything,
)
from .data import class_balanced_weights, stratified_kfold, stratified_split
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
from .losses import tversky_index, tversky_loss
from .uncertainty import risk_coverage_curve
from .metric_learning import supervised_contrastive_loss
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
    clip_by_global_norm,
    EarlyStoppingState,
    RunningAverage,
    batch_indices,
    iter_minibatches,
)
from .history import BestMetric, EpochRecord, MetricHistory
from .losses import binary_focal_loss, dice_coefficient, dice_loss, multiclass_focal_loss
from .calibration import CalibrationBin, TemperatureScaler, calibrated_probabilities, negative_log_likelihood, reliability_bins
from .uncertainty import UncertaintySummary, ensemble_mean, entropy, mutual_information, predictive_entropy, summarize_ensemble, variation_ratio
from .metric_learning import batch_hard_triplet_loss, contrastive_loss, pairwise_distance_matrix, triplet_margin_loss
from .sequences import causal_mask, pad_sequences, sliding_windows, temporal_split
from .regularization import clip_vector_norm, elastic_net_penalty, l1_penalty, l2_penalty, weight_decay_update
from .optimizers import Adam, SGD
from .augmentation import (
    add_gaussian_noise,
    compose_augmentations,
    mixup,
    random_feature_dropout,
    inverted_feature_dropout,
)
from .sequences import lengths_to_padding_mask
from .regularization import group_lasso_penalty
from .validation import (
    bootstrap_mean_interval,
    fold_class_counts,
    kfold_indices,
    stratified_kfold_indices,
)
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
    "tversky_index",
    "tversky_loss",
    "risk_coverage_curve",
    "supervised_contrastive_loss",
    "ndcg_at_k",
    "precision_at_k",
    "brier_score",
    "brier_score_batch",
    "categorical_cross_entropy",
    "categorical_cross_entropy_batch",
    "bootstrap_mean_interval",
    "class_balanced_weights",
    "stratified_kfold",
    "stratified_split",
    "warmup_cosine_decay",
    "huber_loss",
    "huber_loss_batch",
    "mean_absolute_error",
    "mean_squared_error",
    "r2_score",
    "label_smoothed_targets",
    "lengths_to_padding_mask",
    "group_lasso_penalty",
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
    "clip_by_global_norm",
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
    "CalibrationBin",
    "TemperatureScaler",
    "calibrated_probabilities",
    "negative_log_likelihood",
    "reliability_bins",
    "UncertaintySummary",
    "ensemble_mean",
    "entropy",
    "mutual_information",
    "predictive_entropy",
    "summarize_ensemble",
    "variation_ratio",
    "batch_hard_triplet_loss",
    "contrastive_loss",
    "pairwise_distance_matrix",
    "triplet_margin_loss",
    "causal_mask",
    "pad_sequences",
    "sliding_windows",
    "temporal_split",
    "clip_vector_norm",
    "elastic_net_penalty",
    "l1_penalty",
    "l2_penalty",
    "weight_decay_update",
    "Adam",
    "SGD",
]
