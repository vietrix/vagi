"""Backward-compatible AGI config exports."""

from .agi.config import (
    AGIConfig,
    load_agi_config,
    load_agi_large_config,
    load_agi_small_config,
)

__all__ = [
    "AGIConfig",
    "load_agi_config",
    "load_agi_large_config",
    "load_agi_small_config",
]
