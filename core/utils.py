"""Utility helpers for vAGI."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence
import random
from contextlib import contextmanager
import time

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


def validate_seq_len(tensor: torch.Tensor, max_len: int, name: str = "input_ids") -> None:
    seq_len = tensor.shape[1]
    if seq_len <= 0:
        raise ValueError(f"{name} sequence length must be > 0")
    if seq_len > max_len:
        raise ValueError(f"{name} sequence length {seq_len} exceeds max_seq_len={max_len}")


def sanitize_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    clamp_min: float = -1e4,
    clamp_max: float = 1e4,
) -> torch.Tensor:
    """Replace NaN/Inf with finite values and clamp to a safe range."""
    if not tensor.is_floating_point():
        return tensor
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=clamp_max, neginf=clamp_min)
    if clamp_min is not None and clamp_max is not None:
        tensor = torch.clamp(tensor, min=clamp_min, max=clamp_max)
    if not torch.onnx.is_in_onnx_export():
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError(f"{name} contains NaN/Inf after sanitization")
    return tensor


class StageTimer:
    """Track per-stage wall-clock time."""

    def __init__(self) -> None:
        self.times: dict[str, float] = {}

    @contextmanager
    def track(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] = self.times.get(name, 0.0) + (time.perf_counter() - start)


def set_seed(seed: int) -> None:
    """Set RNG seeds for deterministic tests."""
    random.seed(seed)
    torch.manual_seed(seed)
