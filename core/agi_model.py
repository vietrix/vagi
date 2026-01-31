"""Backward-compatible AGI model exports."""

from .agi.model import AGIModel
from .agi.executor import AGIExecutor

__all__ = ["AGIModel", "AGIExecutor"]
