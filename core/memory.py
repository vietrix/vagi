"""Recurrent memory structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class KVCache:
    """Simple KV cache placeholder for future attention caching."""

    keys: Optional[List[torch.Tensor]] = None
    values: Optional[List[torch.Tensor]] = None

    def to(self, device: torch.device | str) -> "KVCache":
        if self.keys is not None:
            self.keys = [k.to(device) for k in self.keys]
        if self.values is not None:
            self.values = [v.to(device) for v in self.values]
        return self


@dataclass
class RecurrentState:
    """Recurrent state for vAGI."""

    mem: torch.Tensor
    kv: KVCache
    timestep: int

    def to(self, device: torch.device | str) -> "RecurrentState":
        return RecurrentState(mem=self.mem.to(device), kv=self.kv.to(device), timestep=self.timestep)
