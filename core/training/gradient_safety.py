"""Gradient safety utilities for stable training.

This module provides:
1. NaN/Inf detection and handling
2. Gradient clipping with various strategies
3. Loss scaling for mixed precision
4. Gradient accumulation helpers
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import logging

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


@dataclass
class GradientSafetyConfig:
    """Configuration for gradient safety mechanisms."""
    # NaN/Inf handling
    check_nan: bool = True
    check_inf: bool = True
    nan_replacement: float = 0.0

    # Gradient clipping
    max_grad_norm: float = 1.0
    clip_type: str = "norm"  # "norm", "value", "adaptive"

    # Adaptive clipping
    adaptive_clip_percentile: float = 0.95
    adaptive_clip_history_size: int = 100

    # Loss scaling
    use_loss_scaling: bool = True
    initial_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000

    # Gradient accumulation
    accumulation_steps: int = 1

    # Recovery
    max_consecutive_nan: int = 3
    recovery_lr_factor: float = 0.1


class GradientMonitor:
    """Monitor gradient statistics and detect anomalies."""

    def __init__(self, config: GradientSafetyConfig):
        self.config = config
        self.grad_history: List[float] = []
        self.nan_count = 0
        self.inf_count = 0
        self.consecutive_nan = 0

    def check_gradients(self, model: nn.Module) -> Dict[str, any]:
        """Check gradients for NaN/Inf values.

        Returns:
            Dict with gradient statistics and any detected issues
        """
        stats = {
            "has_nan": False,
            "has_inf": False,
            "grad_norm": 0.0,
            "max_grad": 0.0,
            "min_grad": 0.0,
            "nan_params": [],
            "inf_params": [],
        }

        total_norm = 0.0
        max_grad = float("-inf")
        min_grad = float("inf")

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.data

            # Check for NaN
            if self.config.check_nan and torch.isnan(grad).any():
                stats["has_nan"] = True
                stats["nan_params"].append(name)
                self.nan_count += 1
                self.consecutive_nan += 1

            # Check for Inf
            if self.config.check_inf and torch.isinf(grad).any():
                stats["has_inf"] = True
                stats["inf_params"].append(name)
                self.inf_count += 1

            # Compute statistics (ignoring NaN/Inf)
            valid_grad = grad[~torch.isnan(grad) & ~torch.isinf(grad)]
            if valid_grad.numel() > 0:
                param_norm = valid_grad.norm().item()
                total_norm += param_norm ** 2
                max_grad = max(max_grad, valid_grad.max().item())
                min_grad = min(min_grad, valid_grad.min().item())

        stats["grad_norm"] = total_norm ** 0.5
        stats["max_grad"] = max_grad if max_grad != float("-inf") else 0.0
        stats["min_grad"] = min_grad if min_grad != float("inf") else 0.0

        # Reset consecutive nan counter if no nan found
        if not stats["has_nan"]:
            self.consecutive_nan = 0

        # Store history for adaptive clipping
        if stats["grad_norm"] > 0:
            self.grad_history.append(stats["grad_norm"])
            if len(self.grad_history) > self.config.adaptive_clip_history_size:
                self.grad_history.pop(0)

        return stats

    def should_skip_update(self, stats: Dict) -> bool:
        """Determine if gradient update should be skipped."""
        if stats["has_nan"] or stats["has_inf"]:
            logger.warning(
                f"Gradient anomaly detected: NaN={stats['has_nan']}, Inf={stats['has_inf']}"
            )
            return True

        if self.consecutive_nan >= self.config.max_consecutive_nan:
            logger.error(
                f"Too many consecutive NaN gradients ({self.consecutive_nan}). "
                "Consider reducing learning rate."
            )
            return True

        return False

    def get_adaptive_clip_value(self) -> float:
        """Get adaptive gradient clip value based on history."""
        if len(self.grad_history) < 10:
            return self.config.max_grad_norm

        sorted_norms = sorted(self.grad_history)
        percentile_idx = int(len(sorted_norms) * self.config.adaptive_clip_percentile)
        return sorted_norms[percentile_idx]


class GradientClipper:
    """Apply gradient clipping with various strategies."""

    def __init__(self, config: GradientSafetyConfig):
        self.config = config
        self.monitor = GradientMonitor(config)

    def clip_gradients(
        self,
        model: nn.Module,
        clip_value: Optional[float] = None
    ) -> float:
        """Clip gradients using configured strategy.

        Args:
            model: Model with gradients to clip
            clip_value: Override clip value (uses config if None)

        Returns:
            Total gradient norm before clipping
        """
        if clip_value is None:
            if self.config.clip_type == "adaptive":
                clip_value = self.monitor.get_adaptive_clip_value()
            else:
                clip_value = self.config.max_grad_norm

        parameters = [p for p in model.parameters() if p.grad is not None]

        if self.config.clip_type == "value":
            # Clip by value
            for p in parameters:
                p.grad.data.clamp_(-clip_value, clip_value)
            total_norm = sum(p.grad.data.norm() ** 2 for p in parameters) ** 0.5
        else:
            # Clip by norm (default)
            total_norm = nn.utils.clip_grad_norm_(parameters, clip_value)

        return total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm


class SafeGradientAccumulator:
    """Safely accumulate gradients with NaN checking."""

    def __init__(self, config: GradientSafetyConfig):
        self.config = config
        self.step_count = 0
        self.accumulated_loss = 0.0

    def should_update(self) -> bool:
        """Check if accumulated enough steps for update."""
        return self.step_count >= self.config.accumulation_steps

    def accumulate(self, loss: torch.Tensor) -> torch.Tensor:
        """Accumulate loss with scaling."""
        scaled_loss = loss / self.config.accumulation_steps
        self.accumulated_loss += loss.item()
        self.step_count += 1
        return scaled_loss

    def reset(self) -> float:
        """Reset accumulator and return accumulated loss."""
        avg_loss = self.accumulated_loss / max(self.step_count, 1)
        self.step_count = 0
        self.accumulated_loss = 0.0
        return avg_loss


class LossScaler:
    """Dynamic loss scaling for mixed precision training."""

    def __init__(self, config: GradientSafetyConfig):
        self.config = config
        self.scale = config.initial_scale
        self.growth_tracker = 0
        self.consecutive_inf = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision."""
        if not self.config.use_loss_scaling:
            return loss
        return loss * self.scale

    def unscale_gradients(self, model: nn.Module) -> bool:
        """Unscale gradients and check for overflow.

        Returns:
            True if gradients are valid, False if overflow detected
        """
        if not self.config.use_loss_scaling:
            return True

        found_inf = False
        inv_scale = 1.0 / self.scale

        for param in model.parameters():
            if param.grad is None:
                continue

            param.grad.data.mul_(inv_scale)

            if torch.isinf(param.grad.data).any():
                found_inf = True

        return not found_inf

    def update_scale(self, overflow: bool):
        """Update scale based on overflow status."""
        if not self.config.use_loss_scaling:
            return

        if overflow:
            self.scale *= self.config.backoff_factor
            self.growth_tracker = 0
            self.consecutive_inf += 1
            logger.warning(f"Gradient overflow, reducing scale to {self.scale}")
        else:
            self.consecutive_inf = 0
            self.growth_tracker += 1

            if self.growth_tracker >= self.config.growth_interval:
                self.scale *= self.config.growth_factor
                self.growth_tracker = 0
                logger.info(f"Increasing scale to {self.scale}")


class GradientSafetyManager:
    """Unified manager for all gradient safety features."""

    def __init__(self, config: Optional[GradientSafetyConfig] = None):
        self.config = config or GradientSafetyConfig()
        self.monitor = GradientMonitor(self.config)
        self.clipper = GradientClipper(self.config)
        self.accumulator = SafeGradientAccumulator(self.config)
        self.scaler = LossScaler(self.config)

    def process_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Process loss with scaling and accumulation."""
        # Check for NaN loss
        if torch.isnan(loss):
            logger.error("NaN loss detected!")
            return torch.tensor(0.0, device=loss.device, requires_grad=True)

        # Scale for mixed precision
        scaled_loss = self.scaler.scale_loss(loss)

        # Accumulate
        return self.accumulator.accumulate(scaled_loss)

    def process_gradients(self, model: nn.Module) -> Tuple[bool, Dict]:
        """Process gradients after backward pass.

        Returns:
            (should_update, gradient_stats)
        """
        # Check for accumulated enough steps
        if not self.accumulator.should_update():
            return False, {}

        # Unscale gradients
        valid = self.scaler.unscale_gradients(model)

        # Check gradient statistics
        stats = self.monitor.check_gradients(model)

        # Update scaler
        overflow = not valid or stats["has_nan"] or stats["has_inf"]
        self.scaler.update_scale(overflow)

        # Determine if should skip update
        if self.monitor.should_skip_update(stats):
            self._zero_invalid_gradients(model)
            return False, stats

        # Clip gradients
        stats["clipped_norm"] = self.clipper.clip_gradients(model)

        return True, stats

    def _zero_invalid_gradients(self, model: nn.Module):
        """Zero out invalid gradients."""
        for param in model.parameters():
            if param.grad is None:
                continue
            mask = torch.isnan(param.grad) | torch.isinf(param.grad)
            param.grad.data[mask] = self.config.nan_replacement

    def step_complete(self):
        """Call after optimizer step."""
        self.accumulator.reset()

    def get_stats(self) -> Dict:
        """Get accumulated statistics."""
        return {
            "total_nan": self.monitor.nan_count,
            "total_inf": self.monitor.inf_count,
            "current_scale": self.scaler.scale,
            "grad_history_len": len(self.monitor.grad_history),
        }


def safe_backward(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    safety_manager: GradientSafetyManager,
    retain_graph: bool = False
) -> Tuple[bool, Dict]:
    """Safe backward pass with all gradient safety features.

    Args:
        loss: Loss to backpropagate
        model: Model to update
        optimizer: Optimizer
        safety_manager: Gradient safety manager
        retain_graph: Whether to retain computation graph

    Returns:
        (success, stats) tuple
    """
    # Process loss
    processed_loss = safety_manager.process_loss(loss)

    # Backward pass
    processed_loss.backward(retain_graph=retain_graph)

    # Process gradients
    should_update, stats = safety_manager.process_gradients(model)

    if should_update:
        optimizer.step()
        optimizer.zero_grad()
        safety_manager.step_complete()
        return True, stats
    else:
        optimizer.zero_grad()
        return False, stats


def check_model_health(model: nn.Module) -> Dict[str, any]:
    """Check overall model health including weights."""
    stats = {
        "has_nan_weights": False,
        "has_inf_weights": False,
        "weight_norm": 0.0,
        "nan_layers": [],
        "inf_layers": [],
    }

    total_norm = 0.0

    for name, param in model.named_parameters():
        if torch.isnan(param.data).any():
            stats["has_nan_weights"] = True
            stats["nan_layers"].append(name)

        if torch.isinf(param.data).any():
            stats["has_inf_weights"] = True
            stats["inf_layers"].append(name)

        valid_weights = param.data[~torch.isnan(param.data) & ~torch.isinf(param.data)]
        if valid_weights.numel() > 0:
            total_norm += valid_weights.norm().item() ** 2

    stats["weight_norm"] = total_norm ** 0.5

    return stats
