"""Observation encoding for the code environment."""

from __future__ import annotations

import hashlib
from typing import Iterable

import torch


def text_to_obs(text: str, obs_dim: int, extra: Iterable[float] | None = None) -> torch.Tensor:
    if obs_dim <= 0:
        raise ValueError("obs_dim must be > 0")
    vec = torch.zeros(obs_dim, dtype=torch.float32)
    tokens = text.split()
    for tok in tokens:
        digest = hashlib.md5(tok.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % obs_dim
        vec[idx] += 1.0
    if extra:
        for i, value in enumerate(extra):
            if i >= obs_dim:
                break
            vec[i] += float(value)
    return vec
