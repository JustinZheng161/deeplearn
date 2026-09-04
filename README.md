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

## Development

Run the test suite with:

```bash
python -m pytest
```
