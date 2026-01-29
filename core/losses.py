"""Loss functions for vAGI."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch.nn import functional as F


def language_loss(
    text_logits: torch.Tensor,
    labels: torch.Tensor,
    k_prefix: int = 0,
    k_suffix: int = 0,
    ignore_index: int = -100,
) -> torch.Tensor:
    if labels.dtype != torch.long:
        raise TypeError("labels must be torch.long")
    if k_prefix < 0:
        raise ValueError("k_prefix must be >= 0")
    if k_suffix < 0:
        raise ValueError("k_suffix must be >= 0")
    end = text_logits.shape[1] - k_suffix if k_suffix else text_logits.shape[1]
    logits = text_logits[:, k_prefix:end, :]
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


def gaussian_nll_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    clamp: Tuple[float, float] = (-10.0, 10.0),
) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=clamp[0], max=clamp[1])
    inv_var = torch.exp(-logvar)
    return 0.5 * torch.mean((target - mean) ** 2 * inv_var + logvar)


def value_loss(value: torch.Tensor, value_targets: torch.Tensor, logvar: Optional[torch.Tensor] = None) -> torch.Tensor:
    if logvar is None:
        return F.mse_loss(value, value_targets)
    return gaussian_nll_loss(value, logvar, value_targets)


def world_loss(
    world_pred: torch.Tensor,
    obs_next: torch.Tensor,
    logvar: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if world_pred.ndim == 3 and obs_next.ndim == 2:
        mean = world_pred[:, 0, :]
    else:
        mean = world_pred
    if logvar is None:
        return F.mse_loss(mean, obs_next)
    if logvar.ndim == 3 and obs_next.ndim == 2:
        logvar = logvar[:, 0, :]
    return gaussian_nll_loss(mean, logvar, obs_next)


def total_loss(losses: Dict[str, torch.Tensor], weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
    weights = weights or {}
    total = None
    for name, loss in losses.items():
        weight = weights.get(name, 1.0)
        total = loss * weight if total is None else total + loss * weight
    if total is None:
        raise ValueError("No losses to combine")
    return total


def consistency_loss(values: torch.Tensor, anchor: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Penalize drift of predicted values across imagined steps."""
    if values.ndim == 2:
        values = values.unsqueeze(-1)
    if values.ndim != 3:
        raise ValueError("values must have shape (B, K, 1) or (B, K)")
    if anchor is None:
        anchor = values[:, 0:1, :]
    if anchor.ndim == 2:
        anchor = anchor.unsqueeze(1)
    if anchor.shape[0] != values.shape[0]:
        raise ValueError("anchor batch size mismatch")
    return F.mse_loss(values, anchor.expand_as(values))


def drift_loss(values: torch.Tensor, max_delta: float = 1.0) -> torch.Tensor:
    """Penalize large step-to-step drift in value predictions."""
    if values.ndim == 2:
        values = values.unsqueeze(-1)
    if values.ndim != 3:
        raise ValueError("values must have shape (B, K, 1) or (B, K)")
    deltas = torch.abs(values[:, 1:, :] - values[:, :-1, :])
    penalty = torch.relu(deltas - max_delta)
    return torch.mean(penalty ** 2)
