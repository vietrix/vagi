"""Dyna-style rollout utilities for vAGI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch.distributions import Categorical

from ..base.memory import KVCache, RecurrentState
from ..training.returns import compute_gae
from ..base.utils import check_floating
from ..base.model import VAGICore


RewardFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: Optional[torch.Tensor] = None

    def validate(self) -> None:
        if self.obs.ndim != 3:
            raise ValueError("obs must have shape (B, T, O)")
        if self.actions.ndim != 2:
            raise ValueError("actions must have shape (B, T)")
        if self.rewards.ndim != 2:
            raise ValueError("rewards must have shape (B, T)")
        if self.dones.ndim != 2:
            raise ValueError("dones must have shape (B, T)")
        if self.obs.shape[0] != self.actions.shape[0]:
            raise ValueError("batch size mismatch in rollout")
        if self.actions.shape != self.rewards.shape or self.actions.shape != self.dones.shape:
            raise ValueError("rollout sequence length mismatch")
        if self.values is not None and self.values.shape[1] != self.actions.shape[1] + 1:
            raise ValueError("values must have shape (B, T + 1)")


def _repeat_state(state: RecurrentState, repeats: int) -> RecurrentState:
    mem = state.mem.repeat_interleave(repeats, dim=0)
    kv = state.kv
    if kv.keys is None or kv.values is None:
        return RecurrentState(mem=mem, kv=kv.clone(), timestep=state.timestep)
    keys = []
    values = []
    for key in kv.keys:
        if key is None:
            keys.append(None)
        else:
            keys.append(key.repeat_interleave(repeats, dim=0))
    for value in kv.values:
        if value is None:
            values.append(None)
        else:
            values.append(value.repeat_interleave(repeats, dim=0))
    return RecurrentState(mem=mem, kv=KVCache(keys=keys, values=values, max_len=kv.max_len), timestep=state.timestep)


def imagine_rollouts(
    model: VAGICore,
    obs: torch.Tensor,
    state: RecurrentState,
    *,
    horizon: int,
    num_rollouts: int = 1,
    task_ids: Optional[torch.Tensor] = None,
    start_token_id: int = 0,
    temperature: float = 1.0,
    deterministic: bool = False,
    reward_fn: Optional[RewardFn] = None,
) -> RolloutBatch:
    if model.world is None:
        raise ValueError("world model is required for imagination rollouts")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if num_rollouts <= 0:
        raise ValueError("num_rollouts must be > 0")
    check_floating(obs, "obs")

    if num_rollouts > 1:
        obs = obs.repeat_interleave(num_rollouts, dim=0)
        state = _repeat_state(state, num_rollouts)
        if task_ids is not None:
            task_ids = task_ids.repeat_interleave(num_rollouts, dim=0)

    batch = obs.shape[0]
    obs_seq = []
    actions = []
    rewards = []
    dones = []
    values = []

    input_ids = torch.full((batch, 1), int(start_token_id), device=obs.device, dtype=torch.long)
    for _ in range(horizon):
        with torch.no_grad():
            out = model.step(input_ids=input_ids, obs=obs, state=state, task_ids=task_ids)
        action_logits = out["action_logits"]
        value = out["value"].squeeze(-1)
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            logits = action_logits / max(temperature, 1e-6)
            action = Categorical(logits=logits).sample()

        world_pred = out["world_pred"]
        if world_pred is None:
            raise ValueError("world_pred is required for imagination rollouts")
        if world_pred.ndim == 3:
            next_obs = world_pred[:, 0, :]
        else:
            next_obs = world_pred

        if reward_fn is None:
            reward = torch.zeros(batch, device=obs.device, dtype=obs.dtype)
            done = torch.zeros(batch, device=obs.device, dtype=obs.dtype)
        else:
            result = reward_fn(obs, action, next_obs)
            if isinstance(result, tuple):
                reward, done = result
            else:
                reward = result
                done = torch.zeros_like(reward)

        obs_seq.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        values.append(value)

        state = out["state"]
        obs = next_obs
        input_ids = action.unsqueeze(1)

    obs_tensor = torch.stack(obs_seq, dim=1)
    actions_tensor = torch.stack(actions, dim=1)
    rewards_tensor = torch.stack(rewards, dim=1)
    dones_tensor = torch.stack(dones, dim=1)
    values_tensor = torch.stack(values, dim=1)
    values_tensor = torch.cat([values_tensor, values_tensor[:, -1:].clone()], dim=1)

    batch_out = RolloutBatch(
        obs=obs_tensor,
        actions=actions_tensor,
        rewards=rewards_tensor,
        dones=dones_tensor,
        values=values_tensor,
    )
    batch_out.validate()
    return batch_out


def mix_rollouts(real: RolloutBatch, imagined: RolloutBatch, *, imagine_ratio: float) -> RolloutBatch:
    real.validate()
    imagined.validate()
    if not (0.0 <= imagine_ratio <= 1.0):
        raise ValueError("imagine_ratio must be in [0, 1]")
    if imagine_ratio == 0.0:
        return real
    if real.obs.shape[1] != imagined.obs.shape[1]:
        raise ValueError("rollouts must have the same horizon to mix")

    num_imagined = max(1, int(real.obs.shape[0] * imagine_ratio))
    idx = torch.randperm(imagined.obs.shape[0], device=imagined.obs.device)[:num_imagined]

    obs = torch.cat([real.obs, imagined.obs[idx]], dim=0)
    actions = torch.cat([real.actions, imagined.actions[idx]], dim=0)
    rewards = torch.cat([real.rewards, imagined.rewards[idx]], dim=0)
    dones = torch.cat([real.dones, imagined.dones[idx]], dim=0)
    values = None
    if real.values is not None and imagined.values is not None:
        values = torch.cat([real.values, imagined.values[idx]], dim=0)

    mixed = RolloutBatch(obs=obs, actions=actions, rewards=rewards, dones=dones, values=values)
    mixed.validate()
    return mixed


def policy_value_losses(
    model: VAGICore,
    rollouts: RolloutBatch,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
    entropy_coef: float = 0.0,
    start_token_id: int = 0,
    task_ids: Optional[torch.Tensor] = None,
    state: Optional[RecurrentState] = None,
) -> dict[str, torch.Tensor]:
    rollouts.validate()
    batch, horizon = rollouts.actions.shape
    if rollouts.values is None:
        raise ValueError("rollouts.values required for policy/value losses")

    advantages, returns = compute_gae(rollouts.rewards, rollouts.values, rollouts.dones, gamma=gamma, lam=lam)
    advantages = advantages.detach()
    returns = returns.detach()

    if state is None:
        state = model.init_state(batch_size=batch, device=rollouts.obs.device)

    step_fn = getattr(VAGICore.step, "__wrapped__", None)
    if step_fn is None:
        raise RuntimeError("VAGICore.step is missing __wrapped__ for grad-enabled use.")

    policy_losses = []
    value_losses = []
    entropy_losses = []
    input_ids = torch.full((batch, 1), int(start_token_id), device=rollouts.obs.device, dtype=torch.long)
    for t in range(horizon):
        obs_t = rollouts.obs[:, t, :]
        out = step_fn(model, input_ids=input_ids, obs=obs_t, state=state, task_ids=task_ids)
        action_logits = out["action_logits"]
        value_pred = out["value"].squeeze(-1)
        dist = Categorical(logits=action_logits)
        actions = rollouts.actions[:, t]
        log_prob = dist.log_prob(actions)
        policy_losses.append(-(log_prob * advantages[:, t]).mean())
        value_losses.append(torch.mean((value_pred - returns[:, t]) ** 2))
        if entropy_coef != 0.0:
            entropy_losses.append(-dist.entropy().mean())

        state = out["state"]
        input_ids = actions.unsqueeze(1)

    policy_loss = torch.stack(policy_losses).mean()
    value_loss = torch.stack(value_losses).mean()
    entropy_loss = torch.stack(entropy_losses).mean() if entropy_losses else torch.tensor(0.0, device=policy_loss.device)
    total = policy_loss + value_loss + entropy_coef * entropy_loss
    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "total_loss": total,
    }


def dyna_update(
    model: VAGICore,
    optimizer: torch.optim.Optimizer,
    real_rollouts: RolloutBatch,
    *,
    imagine_ratio: float,
    imagine_horizon: int,
    num_imagined: int,
    obs: torch.Tensor,
    state: RecurrentState,
    task_ids: Optional[torch.Tensor] = None,
    gamma: float = 0.99,
    lam: float = 0.95,
    entropy_coef: float = 0.0,
    start_token_id: int = 0,
    temperature: float = 1.0,
    deterministic: bool = False,
    reward_fn: Optional[RewardFn] = None,
) -> dict[str, torch.Tensor]:
    """Run a single Dyna-style update with mixed real and imagined rollouts."""
    imagined = imagine_rollouts(
        model,
        obs=obs,
        state=state,
        horizon=imagine_horizon,
        num_rollouts=num_imagined,
        task_ids=task_ids,
        start_token_id=start_token_id,
        temperature=temperature,
        deterministic=deterministic,
        reward_fn=reward_fn,
    )
    mixed = mix_rollouts(real_rollouts, imagined, imagine_ratio=imagine_ratio)
    losses = policy_value_losses(
        model,
        mixed,
        gamma=gamma,
        lam=lam,
        entropy_coef=entropy_coef,
        start_token_id=start_token_id,
        task_ids=task_ids,
    )
    optimizer.zero_grad()
    losses["total_loss"].backward()
    optimizer.step()
    return losses
