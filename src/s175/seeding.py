"""Deterministic seeding.

The original experiments set only np.random.seed and torch.manual_seed, which
leaves CUDA kernel selection, cuDNN algorithm choice and DataLoader worker
seeding non-deterministic. This module sets everything, so that a run can be
reproduced bit-for-bit on the same hardware.
"""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG that affects a training run.

    Call before constructing datasets, models or loaders.

    deterministic=True forces cuDNN into deterministic algorithm selection and
    raises if a non-deterministic kernel is reached. It costs some throughput;
    set False for the timing pilot and True for the recorded runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required by torch for deterministic reductions on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def rng(seed: int) -> np.random.Generator:
    """A local Generator, for anything that should not consume global RNG state."""
    return np.random.default_rng(seed)


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a distinct, reproducible seed."""
    import torch

    base = torch.initial_seed() % (2**31)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)
