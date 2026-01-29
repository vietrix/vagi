"""Utility helpers for training."""

from __future__ import annotations

import os
import random
from typing import Iterable

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)


def set_deterministic(seed: int, deterministic: bool) -> None:
    set_seed(seed)
    if not deterministic:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    for group in optimizer.param_groups:
        return float(group["lr"])
    return 0.0


def iter_batches(dataloader: Iterable):
    for batch in dataloader:
        yield batch
