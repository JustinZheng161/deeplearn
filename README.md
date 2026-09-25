# deeplearn-utils

A small, dependency-light foundation for reproducible deep-learning experiments.

## Reproducibility helpers

`seed_everything` seeds Python and, when installed, NumPy and PyTorch. It returns a `SeedReport` so training code can record which optional backends were available. Pass `deterministic=True` when repeatability is more important than throughput.

```python
from deeplearn_utils import seed_everything

report = seed_everything(42, deterministic=True)
print(report)
```

For a PyTorch `DataLoader`, derive independent streams for workers with `make_worker_init_fn(42)`. The initializer seeds Python, NumPy, and PyTorch consistently when those libraries are installed. Use `derive_seed(42, stream_id)` when you need stable seeds for distributed ranks or other independent consumers. A seeded PyTorch `Generator` is available through `make_generator(42)` when PyTorch is installed.

## Learning-rate scheduling

`warmup_cosine_decay` provides a framework-agnostic learning-rate schedule with linear warmup followed by cosine decay. It returns plain floats and clamps steps after training to the configured minimum:

```python
from deeplearn_utils import warmup_cosine_decay

learning_rate = warmup_cosine_decay(
    step, warmup_steps=500, total_steps=10_000, peak_lr=3e-4, min_lr=1e-6
)
```

## Dataset splitting

`stratified_kfold` creates reproducible cross-validation folds while preserving class proportions in each validation set:

```python
from deeplearn_utils import stratified_kfold

for train_indices, validation_indices in stratified_kfold(labels, folds=5, seed=42):
    train_model(train_indices, validation_indices)
```


`stratified_split` creates reproducible train and validation indices while preserving class representation. It keeps singleton classes in training rather than creating an unusable validation-only sample:

```python
from deeplearn_utils import stratified_split

train_indices, validation_indices = stratified_split(labels, validation_fraction=0.2, seed=42)
```

## Retrieval ranking metrics

For embedding retrieval, `precision_at_k` measures the fraction of top-k results that are relevant, while `ndcg_at_k` rewards relevant results appearing earlier in the ranking. Both complement the existing recall, mean average precision, and mean reciprocal rank helpers:

```python
from deeplearn_utils import ndcg_at_k, precision_at_k

precision = precision_at_k(ranked_ids, relevant_ids, k=10)
ndcg = ndcg_at_k(ranked_ids, relevant_ids, k=10)
```

## Probability calibration

`brier_score` and `brier_score_batch` measure squared probability error for multiclass predictions. Lower scores indicate better probabilistic forecasts, and soft targets are supported alongside the existing expected calibration error helper:

```python
from deeplearn_utils import brier_score

score = brier_score(target_distribution, predicted_probabilities)
```

## Classification losses

`categorical_cross_entropy` and `categorical_cross_entropy_batch` evaluate normalized hard or soft target distributions with stable probability clipping. They work directly with the output of `label_smoothed_targets`:

```python
from deeplearn_utils import categorical_cross_entropy

loss = categorical_cross_entropy(soft_target, predicted_probabilities)
```

## Label smoothing

`label_smoothed_targets` converts integer class labels into probability distributions for soft-target losses. It is framework-agnostic and keeps the target distribution normalized:

```python
from deeplearn_utils import label_smoothed_targets

soft_targets = label_smoothed_targets(labels, class_count=10, smoothing=0.1)
```

## Regression losses

`huber_loss` combines a quadratic penalty for small errors with a linear penalty for large errors, making it useful when regression data contains outliers. `huber_loss_batch` returns the mean over a batch:

```python
from deeplearn_utils import huber_loss_batch

loss = huber_loss_batch(targets, predictions, delta=1.0)
```

## Regression metrics

The package also provides dependency-free `mean_absolute_error`, `mean_squared_error`, and `r2_score` helpers for regression models. Constant targets are handled explicitly so evaluation does not emit an undefined result:

```python
from deeplearn_utils import mean_squared_error, r2_score

mse = mean_squared_error(targets, predictions)
r2 = r2_score(targets, predictions)
```

## Classification metrics

The dependency-free metrics module provides `accuracy`, `balanced_accuracy`, `confusion_matrix`, `precision_recall_f1`, `macro_f1`, and `top_k_accuracy`. They accept ordinary Python sequences, validate shapes and labels, and return JSON-friendly values for experiment reports. `balanced_accuracy` weights each observed class equally, which is useful for imbalanced validation sets.

```python
from deeplearn_utils import macro_f1, top_k_accuracy

f1 = macro_f1([0, 1, 1], [0, 1, 0])
top1 = top_k_accuracy([0, 1], [[0.8, 0.2], [0.4, 0.6]])
```

For neural classifiers whose confidence scores may be overconfident, `expected_calibration_error` measures the confidence-versus-accuracy gap across equal-width probability bins:

```python
from deeplearn_utils import expected_calibration_error

ece = expected_calibration_error(correct, confidences, n_bins=10)
```

## Training loop helpers

`batch_indices` and `iter_minibatches` create reproducible mini-batches without changing the process-wide random generator. `RunningAverage` tracks sample-weighted epoch losses, while `EarlyStopping` monitors validation loss or score and exposes checkpoint-friendly state:

```python
from deeplearn_utils import EarlyStopping, iter_minibatches

stopper = EarlyStopping(patience=3, mode="min")
for features, targets in iter_minibatches(x_train, y_train, 32, shuffle=True, seed=42):
    train_step(features, targets)
if stopper.update(validation_loss):
    print("stop training")
```

## Experiment history

`MetricHistory` records epoch metrics, finds the best loss or score, and serializes cleanly to JSON for checkpoints and experiment artifacts. Missing metrics are represented as `None`, so validation can be logged less frequently than training:

```python
from deeplearn_utils import MetricHistory

history = MetricHistory()
history.add(0, {"loss": 1.2, "accuracy": 0.42})
history.add(1, {"loss": 0.9, "accuracy": 0.61})
best = history.best("loss", mode="min")
checkpoint_metadata = history.to_json(indent=2)
```

## Feature preprocessing

`StandardScaler` and `MinMaxScaler` provide dependency-free, leakage-safe feature transforms. Fit them only on the training split, then reuse the fitted object for validation and test data; constant columns are handled without division errors and both scalers support inverse transforms:

```python
from deeplearn_utils import StandardScaler

scaler = StandardScaler().fit(train_features)
train_scaled = scaler.transform(train_features)
validation_scaled = scaler.transform(validation_features)
```

## Data augmentation

The dependency-free augmentation helpers support reproducible Gaussian noise, feature dropout, and MixUp for numeric batches. `mixup` returns target pairs and interpolation weights so callers can build hard or soft labels in their chosen framework:

```python
from deeplearn_utils import compose_augmentations, mixup

augmented = compose_augmentations(features, noise_stddev=0.01, dropout_probability=0.1, seed=42)
mixed_features, mixed_targets = mixup(features, labels, alpha=0.2, seed=42)
```

## Cross-validation

`kfold_indices` yields complementary train/validation indices, while `stratified_kfold_indices` distributes each class across folds as evenly as possible. Both support deterministic shuffling and keep the original dataset untouched:

```python
from deeplearn_utils import stratified_kfold_indices

for train_indices, validation_indices in stratified_kfold_indices(labels, folds=5, seed=42):
    train_model(train_indices)
    evaluate_model(validation_indices)
```

## Probability utilities

`softmax`, `log_softmax`, and `sigmoid` use numerically stable formulas for converting model logits to probabilities. `categorical_entropy` measures predictive uncertainty, while `binary_cross_entropy` safely clips extreme probabilities:

```python
from deeplearn_utils import categorical_entropy, softmax

probabilities = softmax(logits)
uncertainty = categorical_entropy(probabilities)
```

## Binary decision thresholds

`best_threshold` selects a validation threshold for F1, balanced accuracy, or Youden's J statistic. This is useful when a neural classifier's default `0.5` cutoff is not appropriate for an imbalanced or cost-sensitive task:

```python
from deeplearn_utils import best_threshold, binary_predictions

result = best_threshold(validation_probabilities, validation_labels, objective="f1")
predictions = binary_predictions(test_probabilities, result.threshold)
```

## Embedding retrieval

For representation-learning and nearest-neighbor experiments, the package provides cosine similarity, pairwise similarity matrices, stable ranking, recall@k, mean reciprocal rank, and mean average precision:

```python
from deeplearn_utils import mean_reciprocal_rank, similarity_matrix

scores = similarity_matrix(query_embeddings, candidate_embeddings)
retrieval_mrr = mean_reciprocal_rank(ranked_candidate_ids, relevant_candidate_ids)
```

## Training losses

The package includes framework-agnostic `binary_focal_loss` and `multiclass_focal_loss` helpers for imbalanced classification, plus `dice_coefficient` and `dice_loss` for soft binary-mask overlap. All accept ordinary Python sequences and clip probability boundaries safely:

```python
from deeplearn_utils import binary_focal_loss, dice_loss

classification_loss = binary_focal_loss(labels, positive_probabilities, gamma=2.0)
segmentation_loss = dice_loss(mask, predicted_probabilities)
```

## Development

Run the test suite with:

```bash
python -m pytest
```
