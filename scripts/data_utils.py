"""Dataset helpers for vAGI-core scripts."""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Optional

import torch
from torch.utils.data import Dataset

from vagi_core import VAGIConfig


class TensorDictDataset(Dataset):
    """Dataset backed by a dict of tensors with leading batch dimension."""

    def __init__(self, data: Dict[str, torch.Tensor]) -> None:
        if not data:
            raise ValueError("data must be a non-empty dict of tensors")
        lengths = {tensor.shape[0] for tensor in data.values()}
        if len(lengths) != 1:
            raise ValueError("All tensors must share the same first dimension length")
        self.data = data
        self.length = next(iter(lengths))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {key: tensor[idx] for key, tensor in self.data.items()}


class RandomDataset(Dataset):
    """Deterministic random dataset for quick training runs."""

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
        obs_dim: int,
        action_dim: int,
        with_obs: bool,
        with_world: bool,
        seed: int = 0,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.with_obs = with_obs
        self.with_world = with_world
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gen = torch.Generator()
        gen.manual_seed(self.seed + idx)
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long, generator=gen)
        labels = input_ids.clone()
        batch: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "labels": labels,
            "actions": torch.randint(0, self.action_dim, (1,), dtype=torch.long, generator=gen).squeeze(0),
            "values": torch.randn(1, generator=gen),
        }
        if self.with_obs:
            batch["obs"] = torch.randn(self.obs_dim, generator=gen)
        if self.with_world:
            batch["obs_next"] = torch.randn(self.obs_dim, generator=gen)
        return batch


def load_tensor_dataset(path: str) -> TensorDictDataset:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError("Expected a dict of tensors saved with torch.save")
    if not all(isinstance(v, torch.Tensor) for v in data.values()):
        raise ValueError("All values in the data dict must be torch.Tensor")
    return TensorDictDataset(data)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def validate_batch(batch: Dict[str, torch.Tensor], cfg: VAGIConfig, require_obs: bool) -> None:
    if "input_ids" not in batch:
        raise ValueError("batch missing input_ids")
    if batch["input_ids"].dtype != torch.long:
        raise TypeError("input_ids must be torch.long")
    if "labels" in batch and batch["labels"].dtype != torch.long:
        raise TypeError("labels must be torch.long")
    if require_obs and "obs" not in batch:
        raise ValueError("obs is required but missing from batch")
    if "obs" in batch and batch["obs"].shape[-1] != cfg.obs_dim:
        raise ValueError("obs last dimension must match cfg.obs_dim")
    if "actions" in batch:
        if batch["actions"].ndim == 1:
            if batch["actions"].dtype != torch.long:
                raise TypeError("actions (class indices) must be torch.long")
        elif batch["actions"].ndim == 2:
            if batch["actions"].shape[-1] != cfg.action_dim:
                raise ValueError("actions last dimension must match cfg.action_dim")
        else:
            raise ValueError("actions must be shape (B,) or (B, A)")
    if "values" in batch and batch["values"].shape[-1] != 1:
        raise ValueError("values must have last dimension 1")
    if "obs_next" in batch and batch["obs_next"].shape[-1] != cfg.obs_dim:
        raise ValueError("obs_next last dimension must match cfg.obs_dim")
