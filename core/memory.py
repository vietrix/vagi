"""Recurrent memory structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class KVCache:
    """Simple KV cache placeholder for future attention caching."""

    keys: Optional[List[Optional[torch.Tensor]]] = None
    values: Optional[List[Optional[torch.Tensor]]] = None
    max_len: Optional[int] = None

    @classmethod
    def empty(cls, num_layers: int) -> "KVCache":
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        return cls(keys=[None for _ in range(num_layers)], values=[None for _ in range(num_layers)])

    @classmethod
    def allocate(
        cls,
        num_layers: int,
        batch_size: int,
        n_kv_heads: int,
        head_dim: int,
        max_len: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "KVCache":
        if max_len <= 0:
            raise ValueError("max_len must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be > 0")
        if head_dim <= 0:
            raise ValueError("head_dim must be > 0")
        keys = [
            torch.empty((batch_size, n_kv_heads, 0, head_dim), device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        values = [
            torch.empty((batch_size, n_kv_heads, 0, head_dim), device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        return cls(keys=keys, values=values, max_len=max_len)

    def to(self, device: torch.device | str) -> "KVCache":
        if self.keys is not None:
            self.keys = [k.to(device) if k is not None else None for k in self.keys]
        if self.values is not None:
            self.values = [v.to(device) if v is not None else None for v in self.values]
        return self


@dataclass
class RecurrentState:
    """Recurrent state for vAGI."""

    mem: torch.Tensor
    kv: KVCache
    timestep: int

    def to(self, device: torch.device | str) -> "RecurrentState":
        return RecurrentState(mem=self.mem.to(device), kv=self.kv.to(device), timestep=self.timestep)
