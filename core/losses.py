"""Loss functions for vAGI."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.nn import functional as F


def language_loss(
    text_logits: torch.Tensor, labels: torch.Tensor, k_prefix: int = 0, ignore_index: int = -100
) -> torch.Tensor:
    if labels.dtype != torch.long:
        raise TypeError("labels must be torch.long")
    if k_prefix < 0:
        raise ValueError("k_prefix must be >= 0")
    logits = text_logits[:, k_prefix:, :]
    if logits.shape[1] != labels.shape[1]:
        raise ValueError("labels length must match text token length")
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


def policy_loss(action_logits: torch.Tensor, action_targets: torch.Tensor) -> torch.Tensor:
    if action_targets.dtype == torch.long and action_targets.ndim in (1, 2):
        if action_targets.ndim == 2:
            action_targets = action_targets.squeeze(-1)
        return F.cross_entropy(action_logits, action_targets)
    return F.mse_loss(action_logits, action_targets)


def value_loss(value: torch.Tensor, value_targets: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(value, value_targets)


def world_loss(world_pred: torch.Tensor, obs_next: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(world_pred, obs_next)


def total_loss(losses: Dict[str, torch.Tensor], weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
    weights = weights or {}
    total = None
    for name, loss in losses.items():
        weight = weights.get(name, 1.0)
        total = loss * weight if total is None else total + loss * weight
    if total is None:
        raise ValueError("No losses to combine")
    return total
