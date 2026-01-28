"""Utility helpers for vAGI-core."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence
import random

import torch


def build_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Return a boolean causal mask with True in positions that should be masked."""
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


def check_shape(tensor: torch.Tensor, expected: Sequence[Optional[int]], name: str) -> None:
    """Validate tensor shape against expected dimensions."""
    if tensor.dim() != len(expected):
        raise ValueError(f"{name} has rank {tensor.dim()}, expected {len(expected)}")
    for idx, exp in enumerate(expected):
        if exp is None or exp == -1:
            continue
        if tensor.shape[idx] != exp:
            raise ValueError(f"{name} dim {idx} = {tensor.shape[idx]}, expected {exp}")


def check_floating(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating point tensor")


def set_seed(seed: int) -> None:
    """Set RNG seeds for deterministic tests."""
    random.seed(seed)
    torch.manual_seed(seed)
