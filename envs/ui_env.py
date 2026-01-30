"""Simple UI-like image environment for vision conditioning."""

from __future__ import annotations

import random
from typing import Tuple

import torch


class UIEnv:
    """Deterministic image environment with a target cell action."""

    def __init__(
        self,
        image_size: int = 4,
        action_dim: int | None = None,
        max_steps: int = 32,
        seed: int = 0,
        channels: int = 1,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be > 0")
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if channels <= 0:
            raise ValueError("channels must be > 0")
        self.image_size = image_size
        self.action_dim = action_dim or (image_size * image_size)
        if self.action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        self.max_steps = max_steps
        self.channels = channels
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self.step_count = 0
        self._target = 0
        self._obs = self._make_obs()

    def seed(self, seed: int) -> int:
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        return self._seed

    def reset(self) -> torch.Tensor:
        self.step_count = 0
        self._rng = random.Random(self._seed)
        self._target = self._sample_target()
        self._obs = self._make_obs()
        return self._obs.clone()

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        action_int = int(action)
        target_action = self._target % self.action_dim
        reward = 1.0 if action_int == target_action else 0.0
        self.step_count += 1
        done = self.step_count >= self.max_steps
        self._target = self._sample_target()
        self._obs = self._make_obs()
        info = {"target": int(target_action), "step": self.step_count}
        return self._obs.clone(), reward, done, info

    def render(self) -> None:
        pass

    def _sample_target(self) -> int:
        return self._rng.randrange(self.image_size * self.image_size)

    def _make_obs(self) -> torch.Tensor:
        image = torch.zeros((self.channels, self.image_size, self.image_size), dtype=torch.float32)
        row = self._target // self.image_size
        col = self._target % self.image_size
        image[0, row, col] = 1.0
        if self.channels > 1:
            step_frac = 0.0 if self.max_steps <= 1 else (self.step_count / (self.max_steps - 1))
            image[1:, :, :] = step_frac
        return image
