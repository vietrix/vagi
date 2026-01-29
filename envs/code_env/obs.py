"""Observation encoding for the code environment."""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import torch


def text_to_obs(
    text: str,
    obs_dim: int,
    features: Sequence[float] | None = None,
    feature_slots: int = 8,
) -> torch.Tensor:
    if obs_dim <= 0:
        raise ValueError("obs_dim must be > 0")
    if feature_slots < 0 or feature_slots > obs_dim:
        raise ValueError("feature_slots must be within [0, obs_dim]")
    vec = torch.zeros(obs_dim, dtype=torch.float32)

    if features:
        for i, value in enumerate(features):
            if i >= feature_slots:
                break
            vec[i] = float(value)

    tokens = text.split()
    for tok in tokens:
        digest = hashlib.md5(tok.encode("utf-8")).hexdigest()
        offset = feature_slots
        span = max(1, obs_dim - feature_slots)
        idx = int(digest[:8], 16) % span
        vec[offset + idx] += 1.0
    return vec
