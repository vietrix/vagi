"""Base environment interface."""

from __future__ import annotations

from typing import Protocol, Tuple

import torch


class BaseEnv(Protocol):
    def reset(self) -> torch.Tensor:
        """Reset environment and return initial observation."""

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        """Apply an action and return (obs_next, reward, done, info)."""

    def seed(self, seed: int) -> int:
        """Seed the environment RNG."""
