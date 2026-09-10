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

`stratified_split` creates reproducible train and validation indices while preserving class representation. It keeps singleton classes in training rather than creating an unusable validation-only sample:

```python
from deeplearn_utils import stratified_split

train_indices, validation_indices = stratified_split(labels, validation_fraction=0.2, seed=42)
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

## Development

Run the test suite with:

```bash
python -m pytest
```
