"""Output heads for vAGI-core."""

from __future__ import annotations

import torch
from torch import nn


class LanguageHead(nn.Module):
    """Project hidden states to vocabulary logits."""

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PolicyHead(nn.Module):
    """Project a pooled hidden state to action logits."""

    def __init__(self, hidden_size: int, action_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, action_dim)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        return self.proj(h_last)


class ValueHead(nn.Module):
    """Project a pooled hidden state to a scalar value."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        return self.proj(h_last)


class WorldHead(nn.Module):
    """Project a pooled hidden state to next observation prediction."""

    def __init__(self, hidden_size: int, obs_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, obs_dim)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        return self.proj(h_last)
