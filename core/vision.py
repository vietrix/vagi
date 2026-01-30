"""Minimal vision encoder for image observations."""

from __future__ import annotations

import torch
from torch import nn

from .utils import check_floating, check_shape


class ImageObsEncoder(nn.Module):
    """Encode images into observation vectors."""

    def __init__(self, in_channels: int, obs_dim: int, hidden_size: int = 32) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if obs_dim <= 0:
            raise ValueError("obs_dim must be > 0")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        self.in_channels = in_channels
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(hidden_size, obs_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        check_shape(image, (None, self.in_channels, None, None), "image")
        check_floating(image, "image")
        x = self.conv(image)
        x = self.pool(x).flatten(1)
        return self.proj(x)
