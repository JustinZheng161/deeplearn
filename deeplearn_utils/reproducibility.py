"""Utilities for repeatable deep-learning experiments."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SeedReport:
    """Describe which optional numerical backends were seeded."""

    seed: int
    numpy: bool
    torch: bool


def _optional_numpy():
    try:
        import numpy as np
    except ImportError:
        return None
    return np


def _optional_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def seed_everything(seed: int, *, deterministic: bool = False) -> SeedReport:
    """Seed common experiment sources and return an auditable summary.

    The function does not require NumPy or PyTorch to be installed. When
    ``deterministic`` is true, PyTorch deterministic algorithms are enabled
    and the CUDA cuBLAS workspace setting is configured before CUDA use.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    numpy = _optional_numpy()
    if numpy is not None:
        numpy.random.seed(seed)

    torch = _optional_torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

    return SeedReport(seed=seed, numpy=numpy is not None, torch=torch is not None)


def make_worker_init_fn(seed: int) -> Callable[[int], None]:
    """Create a DataLoader worker initializer with independent child seeds."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    def initialize(worker_id: int) -> None:
        if not isinstance(worker_id, int) or worker_id < 0:
            raise ValueError("worker_id must be a non-negative integer")
        child_seed = (seed + worker_id) % (2**32)
        random.seed(child_seed)
        numpy = _optional_numpy()
        if numpy is not None:
            numpy.random.seed(child_seed)

    return initialize


def make_generator(seed: int):
    """Return a seeded PyTorch generator, or fail clearly when unavailable."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    torch = _optional_torch()
    if torch is None:
        raise RuntimeError("make_generator requires PyTorch to be installed")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
