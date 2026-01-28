"""Utility helpers for training."""

from __future__ import annotations

import random
from typing import Iterable

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    for group in optimizer.param_groups:
        return float(group["lr"])
    return 0.0


def iter_batches(dataloader: Iterable):
    for batch in dataloader:
        yield batch
