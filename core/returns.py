"""Return and advantage utilities for training."""

from __future__ import annotations

from typing import Tuple

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute generalized advantage estimates and returns.

    Expects values to have one extra step for bootstrapping (T + 1).
    """
    if rewards.ndim == 1:
        rewards = rewards.unsqueeze(0)
    if dones.ndim == 1:
        dones = dones.unsqueeze(0)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if rewards.shape != dones.shape:
        raise ValueError("rewards and dones must have the same shape")
    if values.shape[1] != rewards.shape[1] + 1:
        raise ValueError("values must have shape (B, T + 1)")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lam must be in [0, 1]")

    batch_size, horizon = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(batch_size, device=rewards.device, dtype=rewards.dtype)

    for t in range(horizon - 1, -1, -1):
        done = dones[:, t]
        next_value = values[:, t + 1]
        delta = rewards[:, t] + gamma * next_value * (1.0 - done) - values[:, t]
        gae = delta + gamma * lam * (1.0 - done) * gae
        advantages[:, t] = gae

    returns = advantages + values[:, :-1]
    return advantages, returns


def td_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """Compute TD(lambda) returns using GAE-style recursion."""
    _adv, returns = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)
    return returns
