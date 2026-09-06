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

## Dataset splitting

`stratified_split` creates reproducible train and validation indices while preserving class representation. It keeps singleton classes in training rather than creating an unusable validation-only sample:

```python
from deeplearn_utils import stratified_split

train_indices, validation_indices = stratified_split(labels, validation_fraction=0.2, seed=42)
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

## Development

Run the test suite with:

```bash
python -m pytest
```
