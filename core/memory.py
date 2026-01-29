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

    @classmethod
    def empty(cls, num_layers: int) -> "KVCache":
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        return cls(keys=[None for _ in range(num_layers)], values=[None for _ in range(num_layers)])

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
