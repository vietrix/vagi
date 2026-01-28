"""Deterministic toy environment for vAGI."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StepResult:
    obs: torch.Tensor
    reward: float
    done: bool
    info: dict


class ToyEnv:
    """Simple deterministic environment with a 1D position target."""

    def __init__(self, obs_dim: int = 16, action_dim: int = 4, max_steps: int = 20, target: int = 5) -> None:
        if obs_dim < 4:
            raise ValueError("obs_dim must be >= 4")
        if action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.target = target
        self.reset()

    def reset(self) -> torch.Tensor:
        self.position = 0
        self.step_count = 0
        self.last_action = 0
        return self._get_obs()

    def step(self, action: int | torch.Tensor) -> StepResult:
        action_int = int(action) if not isinstance(action, torch.Tensor) else int(action.item())
        self.last_action = action_int
        if action_int == 1:
            self.position += 1
        elif action_int == 2:
            self.position -= 1

        self.step_count += 1
        done = self.step_count >= self.max_steps or self.position == self.target
        reward = 1.0 if self.position == self.target else -0.01
        obs = self._get_obs()
        info = {"position": self.position, "target": self.target, "step": self.step_count}
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def _get_obs(self) -> torch.Tensor:
        obs = torch.zeros(self.obs_dim, dtype=torch.float32)
        obs[0] = float(self.position)
        obs[1] = float(self.target)
        obs[2] = float(self.step_count)
        obs[3] = float(self.last_action)
        return obs
