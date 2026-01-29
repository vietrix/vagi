"""Pack rollout records into batch tensors."""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List

import torch

from .schema import RolloutRecord


def pack_batches(
    records: Iterable[RolloutRecord],
    *,
    batch_size: int,
    horizon: int,
    gamma: float,
    task_id: int | None = None,
    gae_lambda: float | None = None,
) -> Iterator[Dict[str, torch.Tensor]]:
    """Yield packed batches from a stream of rollout records."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")

    buffer: List[Dict[str, torch.Tensor]] = []
    for episode in _iter_episodes(records):
        samples = _episode_to_samples(
            episode, horizon=horizon, gamma=gamma, task_id=task_id, gae_lambda=gae_lambda
        )
        for sample in samples:
            buffer.append(sample)
            if len(buffer) >= batch_size:
                yield _collate(buffer)
                buffer = []

    if buffer:
        yield _collate(buffer)


def _iter_episodes(records: Iterable[RolloutRecord]) -> Iterator[List[RolloutRecord]]:
    current_id = None
    episode: List[RolloutRecord] = []
    for record in records:
        if current_id is None:
            current_id = record.episode_id
        if record.episode_id != current_id and episode:
            yield episode
            episode = []
            current_id = record.episode_id
        episode.append(record)
        if record.done:
            yield episode
            episode = []
    if episode:
        yield episode


def _episode_to_samples(
    episode: List[RolloutRecord],
    *,
    horizon: int,
    gamma: float,
    task_id: int | None,
    gae_lambda: float | None,
) -> List[Dict[str, torch.Tensor]]:
    obs_dim = len(episode[0].obs)
    obs_seq = [record.obs for record in episode]
    rewards = [float(record.reward) for record in episode]
    values = [float(record.value) if record.value is not None else 0.0 for record in episode]
    use_gae = gae_lambda is not None and all(record.value is not None for record in episode)
    if use_gae:
        advantages, returns = _compute_gae(rewards, values, gamma, float(gae_lambda))
    else:
        returns = _compute_returns(rewards, gamma)
        advantages = None
    samples: List[Dict[str, torch.Tensor]] = []

    for idx, record in enumerate(episode):
        future_obs: List[List[float]] = []
        masks: List[float] = []
        for offset in range(1, horizon + 1):
            step_idx = idx + offset
            if step_idx < len(obs_seq):
                future_obs.append(obs_seq[step_idx])
                masks.append(1.0)
            else:
                future_obs.append([0.0 for _ in range(obs_dim)])
                masks.append(0.0)

        sample: Dict[str, torch.Tensor] = {
            "obs": torch.tensor(record.obs, dtype=torch.float32),
            "obs_future": torch.tensor(future_obs, dtype=torch.float32),
            "actions": torch.tensor(record.action, dtype=torch.long),
            "returns": torch.tensor([returns[idx]], dtype=torch.float32),
            "rewards": torch.tensor([record.reward], dtype=torch.float32),
            "mask": torch.tensor(masks, dtype=torch.float32),
        }
        if advantages is not None:
            sample["advantages"] = torch.tensor([advantages[idx]], dtype=torch.float32)
        if task_id is not None:
            sample["task_id"] = torch.tensor(task_id, dtype=torch.long)
        samples.append(sample)
    return samples


def _compute_returns(rewards: List[float], gamma: float) -> List[float]:
    returns = [0.0 for _ in rewards]
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = rewards[idx] + gamma * running
        returns[idx] = running
    return returns


def _collate(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch: Dict[str, torch.Tensor] = {
        "obs": torch.stack([s["obs"] for s in samples], dim=0),
        "obs_future": torch.stack([s["obs_future"] for s in samples], dim=0),
        "actions": torch.stack([s["actions"] for s in samples], dim=0),
        "returns": torch.stack([s["returns"] for s in samples], dim=0),
        "rewards": torch.stack([s["rewards"] for s in samples], dim=0),
        "mask": torch.stack([s["mask"] for s in samples], dim=0),
    }
    if "task_id" in samples[0]:
        batch["task_id"] = torch.stack([s["task_id"] for s in samples], dim=0)
    if "advantages" in samples[0]:
        batch["advantages"] = torch.stack([s["advantages"] for s in samples], dim=0)
    return batch


def _compute_gae(
    rewards: List[float],
    values: List[float],
    gamma: float,
    gae_lambda: float,
) -> tuple[List[float], List[float]]:
    advantages = [0.0 for _ in rewards]
    returns = [0.0 for _ in rewards]
    gae = 0.0
    next_value = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        delta = rewards[idx] + gamma * next_value - values[idx]
        gae = delta + gamma * gae_lambda * gae
        advantages[idx] = gae
        returns[idx] = gae + values[idx]
        next_value = values[idx]
    return advantages, returns
