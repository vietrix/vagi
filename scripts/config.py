"""Configuration for minimal training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    data_path: str
    out_dir: str
    epochs: int = 1
    batch_size: int = 8
    lr: float = 3e-4
    max_steps: Optional[int] = None
    save_every: int = 100
    log_every: int = 10
    grad_accum: int = 1
    seed: int = 0
    max_seq_len: int = 128
    hidden_size: int = 64
    n_layers: int = 2
    n_heads: int = 4
    mlp_ratio: float = 2.0
    obs_dim: int = 0
    obs_tokens: int = 0
    action_dim: int = 8
    memory_slots: int = 4
    use_world_pred: bool = False
