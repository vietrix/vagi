"""Backbone components for vAGI-core."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .config import VAGIConfig
from .memory import RecurrentState
from .utils import build_causal_mask, check_floating, check_shape


class ObsTokenizer(nn.Module):
    """Project observations into a fixed number of tokens."""

    def __init__(self, obs_dim: int, obs_tokens: int, hidden_size: int) -> None:
        super().__init__()
        self.obs_tokens = obs_tokens
        self.hidden_size = hidden_size
        self.proj = nn.Linear(obs_dim, obs_tokens * hidden_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        check_shape(obs, (None, None), "obs")
        check_floating(obs, "obs")
        bsz = obs.shape[0]
        projected = self.proj(obs)
        return projected.view(bsz, self.obs_tokens, self.hidden_size)


class FastMemory(nn.Module):
    """Fast memory update module with gated writes."""

    def __init__(self, hidden_size: int, memory_slots: int) -> None:
        super().__init__()
        self.memory_slots = memory_slots
        self.gate = nn.Linear(hidden_size, memory_slots)
        self.write = nn.Linear(hidden_size, hidden_size)

    def forward(self, mem: torch.Tensor, h_last: torch.Tensor) -> torch.Tensor:
        check_shape(mem, (None, self.memory_slots, None), "mem")
        gate = torch.sigmoid(self.gate(h_last)).unsqueeze(-1)
        write = self.write(h_last).unsqueeze(1)
        return mem + gate * write


class CausalTransformerBackbone(nn.Module):
    """Causal transformer decoder backbone with optional observation tokens."""

    def __init__(self, cfg: VAGIConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_embed = nn.Embedding(cfg.max_seq_len + cfg.obs_tokens, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.obs_tokens = cfg.obs_tokens
        self.obs_tokenizer = ObsTokenizer(cfg.obs_dim, cfg.obs_tokens, cfg.hidden_size) if cfg.obs_tokens > 0 else None
        self.memory = FastMemory(cfg.hidden_size, cfg.memory_slots) if cfg.memory_slots > 0 else None

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_size,
            nhead=cfg.n_heads,
            dim_feedforward=int(cfg.hidden_size * cfg.mlp_ratio),
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=cfg.n_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        obs: Optional[torch.Tensor] = None,
        state: Optional[RecurrentState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must be torch.long")
        check_shape(input_ids, (None, None), "input_ids")

        x = self.token_embed(input_ids)

        if obs is not None and self.obs_tokenizer is not None:
            obs_tokens = self.obs_tokenizer(obs)
            x = torch.cat([obs_tokens, x], dim=1)

        seq_len = x.shape[1]
        if seq_len > self.pos_embed.num_embeddings:
            raise ValueError("Sequence length exceeds configured max_seq_len + obs_tokens")

        pos_ids = torch.arange(seq_len, device=x.device)
        x = x + self.pos_embed(pos_ids).unsqueeze(0)
        x = self.dropout(x)

        mask = build_causal_mask(seq_len, device=x.device)
        x = self.encoder(x, mask=mask)

        h_last = x[:, -1, :]

        mem_next = None
        if state is not None and self.memory is not None:
            mem_next = self.memory(state.mem, h_last)

        return x, h_last, mem_next
