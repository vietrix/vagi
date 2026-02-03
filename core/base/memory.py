"""Recurrent memory structures with efficient KV-cache implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class KVCache:
    """Efficient KV cache with sliding window support for attention.

    Features:
    - Pre-allocated buffers for efficient memory usage
    - Sliding window support for bounded memory
    - Position tracking for rotary embeddings
    - Batch-aware operations
    """

    keys: Optional[List[Optional[torch.Tensor]]] = None
    values: Optional[List[Optional[torch.Tensor]]] = None
    max_len: Optional[int] = None
    sliding_window: Optional[int] = None  # If set, use sliding window attention
    current_len: int = 0  # Track how much of the cache is filled
    start_pos: int = 0  # Starting position for rotary embeddings

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
        sliding_window: Optional[int] = None,
    ) -> "KVCache":
        """Allocate pre-sized KV cache buffers.

        Args:
            num_layers: Number of transformer layers
            batch_size: Batch size
            n_kv_heads: Number of KV heads (for GQA)
            head_dim: Dimension per head
            max_len: Maximum sequence length
            device: Target device
            dtype: Data type
            sliding_window: If set, limit cache to this window size
        """
        if max_len <= 0:
            raise ValueError("max_len must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be > 0")
        if head_dim <= 0:
            raise ValueError("head_dim must be > 0")

        # Determine actual cache size
        cache_size = sliding_window if sliding_window else max_len

        # Pre-allocate full buffers (more efficient than concatenation)
        keys = [
            torch.zeros((batch_size, n_kv_heads, cache_size, head_dim), device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        values = [
            torch.zeros((batch_size, n_kv_heads, cache_size, head_dim), device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        return cls(
            keys=keys,
            values=values,
            max_len=max_len,
            sliding_window=sliding_window,
            current_len=0,
            start_pos=0
        )

    def update(
        self,
        layer_idx: int,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new keys/values and return full cache for attention.

        Args:
            layer_idx: Layer index
            new_keys: New keys [B, n_kv_heads, seq_len, head_dim]
            new_values: New values [B, n_kv_heads, seq_len, head_dim]

        Returns:
            (cached_keys, cached_values) for attention computation
        """
        if self.keys is None or self.values is None:
            return new_keys, new_values

        seq_len = new_keys.size(2)
        cache_size = self.keys[layer_idx].size(2)

        if self.sliding_window:
            # Sliding window: shift old values and append new
            if self.current_len + seq_len > cache_size:
                # Need to shift
                shift = self.current_len + seq_len - cache_size
                self.keys[layer_idx][:, :, :-shift] = self.keys[layer_idx][:, :, shift:].clone()
                self.values[layer_idx][:, :, :-shift] = self.values[layer_idx][:, :, shift:].clone()
                self.current_len = cache_size - seq_len
                self.start_pos += shift

            # Insert new values
            start = self.current_len
            end = start + seq_len
            self.keys[layer_idx][:, :, start:end] = new_keys
            self.values[layer_idx][:, :, start:end] = new_values

            # Return only the filled portion
            return (
                self.keys[layer_idx][:, :, :end],
                self.values[layer_idx][:, :, :end]
            )
        else:
            # Standard cache: append until full
            if self.current_len + seq_len <= cache_size:
                start = self.current_len
                end = start + seq_len
                self.keys[layer_idx][:, :, start:end] = new_keys
                self.values[layer_idx][:, :, start:end] = new_values

                # Only update current_len on first layer to avoid double counting
                if layer_idx == 0:
                    self.current_len = end

                return (
                    self.keys[layer_idx][:, :, :end],
                    self.values[layer_idx][:, :, :end]
                )
            else:
                # Cache full, just return new values (shouldn't happen in practice)
                return new_keys, new_values

    def get_position(self) -> int:
        """Get current position for rotary embeddings."""
        return self.start_pos + self.current_len

    def reset(self):
        """Reset cache to empty state."""
        self.current_len = 0
        self.start_pos = 0
        if self.keys is not None:
            for k in self.keys:
                if k is not None:
                    k.zero_()
        if self.values is not None:
            for v in self.values:
                if v is not None:
                    v.zero_()

    def to(self, device: torch.device | str) -> "KVCache":
        if self.keys is not None:
            self.keys = [k.to(device) if k is not None else None for k in self.keys]
        if self.values is not None:
            self.values = [v.to(device) if v is not None else None for v in self.values]
        return self

    def clone(self) -> "KVCache":
        keys = None
        values = None
        if self.keys is not None:
            keys = [k.clone() if k is not None else None for k in self.keys]
        if self.values is not None:
            values = [v.clone() if v is not None else None for v in self.values]
        return KVCache(
            keys=keys,
            values=values,
            max_len=self.max_len,
            sliding_window=self.sliding_window,
            current_len=self.current_len,
            start_pos=self.start_pos
        )

    def __len__(self) -> int:
        """Return current cache length."""
        return self.current_len


@dataclass
class FastMemory:
    """Fast differentiable memory with improved aggregation methods.

    Supports multiple aggregation strategies:
    - mean: Average all slots
    - max: Max pooling across slots
    - attention: Query-based attention aggregation
    """

    slots: torch.Tensor  # [batch, num_slots, hidden_size]
    write_weights: Optional[torch.Tensor] = None  # For tracking writes
    aggregation_method: str = "mean"

    @classmethod
    def init(
        cls,
        batch_size: int,
        num_slots: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        aggregation_method: str = "mean",
    ) -> "FastMemory":
        """Initialize empty fast memory."""
        slots = torch.zeros(batch_size, num_slots, hidden_size, device=device, dtype=dtype)
        return cls(slots=slots, aggregation_method=aggregation_method)

    def read(self, query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Read from memory with configurable aggregation.

        Args:
            query: Optional query for attention-based reading [batch, hidden_size]

        Returns:
            Aggregated memory content [batch, hidden_size]
        """
        if self.aggregation_method == "max":
            return self.slots.max(dim=1)[0]
        elif self.aggregation_method == "attention" and query is not None:
            # Attention-based aggregation
            scores = torch.einsum("bnh,bh->bn", self.slots, query)
            weights = torch.softmax(scores, dim=1)
            return torch.einsum("bn,bnh->bh", weights, self.slots)
        else:
            # Default: mean
            return self.slots.mean(dim=1)

    def write(
        self,
        content: torch.Tensor,
        slot_idx: Optional[int] = None,
        decay: float = 0.99,
    ) -> "FastMemory":
        """Write to memory with optional decay.

        Args:
            content: Content to write [batch, hidden_size]
            slot_idx: Specific slot to write to (None = all slots with decay)
            decay: Decay factor for existing content

        Returns:
            Updated FastMemory
        """
        if slot_idx is not None:
            new_slots = self.slots.clone()
            new_slots[:, slot_idx] = content
        else:
            # Decay existing and add new content to first slot
            new_slots = self.slots * decay
            new_slots[:, 0] = content
        return FastMemory(
            slots=new_slots,
            aggregation_method=self.aggregation_method
        )

    def consolidate(self) -> "FastMemory":
        """Consolidate memory by moving summary to first slot."""
        summary = self.read()
        new_slots = self.slots.clone()
        # Rotate slots and put summary first
        new_slots[:, 1:] = self.slots[:, :-1]
        new_slots[:, 0] = summary
        return FastMemory(
            slots=new_slots,
            aggregation_method=self.aggregation_method
        )

    def to(self, device: torch.device) -> "FastMemory":
        return FastMemory(
            slots=self.slots.to(device),
            write_weights=self.write_weights.to(device) if self.write_weights is not None else None,
            aggregation_method=self.aggregation_method
        )

    def clone(self) -> "FastMemory":
        return FastMemory(
            slots=self.slots.clone(),
            write_weights=self.write_weights.clone() if self.write_weights is not None else None,
            aggregation_method=self.aggregation_method
        )


@dataclass
class RecurrentState:
    """Recurrent state for vAGI."""

    mem: torch.Tensor
    kv: KVCache
    timestep: int

    def to(self, device: torch.device | str) -> "RecurrentState":
        return RecurrentState(mem=self.mem.to(device), kv=self.kv.to(device), timestep=self.timestep)

    def clone(self) -> "RecurrentState":
        return RecurrentState(mem=self.mem.clone(), kv=self.kv.clone(), timestep=self.timestep)
