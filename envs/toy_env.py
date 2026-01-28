"""Deterministic toy environment for vAGI."""

from __future__ import annotations

import random
from typing import Tuple

import torch


class ToyEnv:
    """Deterministic environment with a rule-based target action."""

    def __init__(self, obs_dim: int = 8, action_dim: int = 4, max_steps: int = 32, seed: int = 0) -> None:
        if obs_dim < action_dim:
            raise ValueError("obs_dim must be >= action_dim")
        if action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self._seed = seed
        self._rng = random.Random(seed)
        self.step_count = 0
        self._obs = self._make_obs()

    def seed(self, seed: int) -> int:
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        return self._seed

    def reset(self) -> torch.Tensor:
        self.step_count = 0
        self._rng = random.Random(self._seed)
        self._obs = self._make_obs()
        return self._obs.clone()

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        action_int = int(action)
        target = self._target_action(self._obs)
        reward = 1.0 if action_int == target else 0.0

        self.step_count += 1
        done = self.step_count >= self.max_steps
        self._obs = self._make_obs()
        info = {"target": target, "step": self.step_count}
        return self._obs.clone(), reward, done, info

    def render(self) -> None:
        pass

    def _make_obs(self) -> torch.Tensor:
        values = [self._rng.random() for _ in range(self.obs_dim)]
        return torch.tensor(values, dtype=torch.float32)

    def _target_action(self, obs: torch.Tensor) -> int:
        slice_vals = obs[: self.action_dim]
        return int(torch.argmax(slice_vals).item())
