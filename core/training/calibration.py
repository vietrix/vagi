"""Confidence calibration utilities with online calibration support.

This module provides:
1. Temperature scaling calibration
2. Platt scaling
3. Online calibration updates
4. ECE (Expected Calibration Error) metrics
5. Integration with metacognition training
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def _safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert probability to logit with numerical safety."""
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


def brier_score(confidence: torch.Tensor, outcomes: torch.Tensor) -> torch.Tensor:
    """Compute Brier score for calibration evaluation."""
    if confidence.shape != outcomes.shape:
        raise ValueError("confidence and outcomes must have the same shape")
    return torch.mean((confidence - outcomes) ** 2)


def expected_calibration_error(
    confidence: torch.Tensor,
    outcomes: torch.Tensor,
    num_bins: int = 10
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute Expected Calibration Error (ECE).

    Args:
        confidence: Predicted confidences [N]
        outcomes: Binary outcomes (0 or 1) [N]
        num_bins: Number of bins for calibration

    Returns:
        ECE value and per-bin statistics
    """
    if confidence.numel() == 0:
        return torch.tensor(0.0), {}

    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=confidence.device)
    ece = torch.tensor(0.0, device=confidence.device)

    bin_stats = {
        "bin_accuracy": [],
        "bin_confidence": [],
        "bin_count": [],
    }

    for i in range(num_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]

        # Find samples in this bin
        in_bin = (confidence > lower) & (confidence <= upper)
        prop_in_bin = in_bin.float().mean()

        if in_bin.sum() > 0:
            accuracy_in_bin = outcomes[in_bin].float().mean()
            avg_confidence_in_bin = confidence[in_bin].mean()

            ece += prop_in_bin * torch.abs(accuracy_in_bin - avg_confidence_in_bin)

            bin_stats["bin_accuracy"].append(accuracy_in_bin.item())
            bin_stats["bin_confidence"].append(avg_confidence_in_bin.item())
            bin_stats["bin_count"].append(in_bin.sum().item())
        else:
            bin_stats["bin_accuracy"].append(0.0)
            bin_stats["bin_confidence"].append(0.0)
            bin_stats["bin_count"].append(0)

    return ece, bin_stats


def maximum_calibration_error(
    confidence: torch.Tensor,
    outcomes: torch.Tensor,
    num_bins: int = 10
) -> torch.Tensor:
    """Compute Maximum Calibration Error (MCE)."""
    if confidence.numel() == 0:
        return torch.tensor(0.0)

    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=confidence.device)
    max_error = torch.tensor(0.0, device=confidence.device)

    for i in range(num_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidence > lower) & (confidence <= upper)

        if in_bin.sum() > 0:
            accuracy = outcomes[in_bin].float().mean()
            avg_conf = confidence[in_bin].mean()
            error = torch.abs(accuracy - avg_conf)
            max_error = torch.max(max_error, error)

    return max_error


@dataclass
class ConfidenceCalibrator:
    temperature: float = 1.0
    bias: float = 0.0

    def apply(self, confidence: torch.Tensor) -> torch.Tensor:
        if confidence.numel() == 0:
            return confidence
        logit = _safe_logit(confidence)
        temp = max(self.temperature, 1e-6)
        return torch.sigmoid(logit / temp + self.bias)

    def fit(
        self,
        confidence: torch.Tensor,
        outcomes: torch.Tensor,
        *,
        steps: int = 200,
        lr: float = 0.1,
    ) -> "ConfidenceCalibrator":
        if confidence.shape != outcomes.shape:
            raise ValueError("confidence and outcomes must have the same shape")
        if steps <= 0:
            raise ValueError("steps must be > 0")
        if lr <= 0:
            raise ValueError("lr must be > 0")

        device = confidence.device
        log_temp = nn.Parameter(torch.tensor(0.0, device=device))
        bias = nn.Parameter(torch.tensor(0.0, device=device))
        optimizer = torch.optim.Adam([log_temp, bias], lr=lr)
        bce = nn.BCEWithLogitsLoss()

        target = outcomes.detach()
        logits = _safe_logit(confidence.detach())

        for _ in range(steps):
            optimizer.zero_grad()
            temp = torch.exp(log_temp)
            calibrated = logits / temp + bias
            loss = bce(calibrated, target)
            loss.backward()
            optimizer.step()

        self.temperature = float(torch.exp(log_temp).item())
        self.bias = float(bias.item())
        return self


class OnlineCalibrator(nn.Module):
    """Online calibration with running updates.

    Maintains a moving window of predictions and outcomes
    for continuous calibration updates during training.
    """

    def __init__(
        self,
        window_size: int = 1000,
        update_interval: int = 100,
        min_samples: int = 50,
    ):
        super().__init__()
        self.window_size = window_size
        self.update_interval = update_interval
        self.min_samples = min_samples

        # Learnable calibration parameters
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

        # Buffer for online updates
        self.register_buffer("confidence_buffer", torch.zeros(window_size))
        self.register_buffer("outcome_buffer", torch.zeros(window_size))
        self.register_buffer("buffer_idx", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_samples", torch.tensor(0, dtype=torch.long))

    def add_observation(
        self,
        confidence: torch.Tensor,
        outcome: torch.Tensor
    ):
        """Add new observation to buffer."""
        # Handle batched inputs
        conf_flat = confidence.flatten()
        out_flat = outcome.flatten()

        for c, o in zip(conf_flat, out_flat):
            idx = self.buffer_idx.item() % self.window_size
            self.confidence_buffer[idx] = c.item()
            self.outcome_buffer[idx] = o.item()
            self.buffer_idx += 1
            self.total_samples += 1

    def calibrate(self, confidence: torch.Tensor) -> torch.Tensor:
        """Apply calibration to confidence values."""
        logit = _safe_logit(confidence)
        temp = torch.exp(self.log_temperature).clamp(min=0.1, max=10.0)
        return torch.sigmoid(logit / temp + self.bias)

    def get_calibration_loss(self) -> torch.Tensor:
        """Compute calibration loss from buffer."""
        n_samples = min(self.total_samples.item(), self.window_size)
        if n_samples < self.min_samples:
            return torch.tensor(0.0, device=self.log_temperature.device)

        conf = self.confidence_buffer[:n_samples]
        outcomes = self.outcome_buffer[:n_samples]

        # Apply current calibration
        calibrated = self.calibrate(conf)

        # BCE loss for calibration
        loss = F.binary_cross_entropy(calibrated, outcomes)

        return loss

    def get_metrics(self) -> Dict[str, float]:
        """Get current calibration metrics."""
        n_samples = min(self.total_samples.item(), self.window_size)
        if n_samples < self.min_samples:
            return {"ece": 0.0, "brier": 0.0, "samples": n_samples}

        conf = self.confidence_buffer[:n_samples]
        outcomes = self.outcome_buffer[:n_samples]

        calibrated = self.calibrate(conf)
        ece, _ = expected_calibration_error(calibrated, outcomes)
        brier = brier_score(calibrated, outcomes)

        return {
            "ece": ece.item(),
            "brier": brier.item(),
            "temperature": torch.exp(self.log_temperature).item(),
            "bias": self.bias.item(),
            "samples": n_samples,
        }


class MetaCognitionTrainer:
    """Training utilities for metacognition module.

    Handles:
    1. Confidence calibration training
    2. Self-assessment accuracy tracking
    3. Integration with main training loop
    """

    def __init__(
        self,
        metacognition_module: nn.Module,
        learning_rate: float = 1e-4,
        calibration_weight: float = 0.1,
        self_assessment_weight: float = 0.1,
    ):
        self.metacog = metacognition_module
        self.calibration_weight = calibration_weight
        self.self_assessment_weight = self_assessment_weight

        # Online calibrator
        self.calibrator = OnlineCalibrator()

        # Optimizer for metacognition
        self.optimizer = torch.optim.Adam(
            list(metacognition_module.parameters()) + list(self.calibrator.parameters()),
            lr=learning_rate
        )

        # Tracking
        self.total_predictions = 0
        self.correct_predictions = 0

    def update(
        self,
        task_embedding: torch.Tensor,
        predicted_success: torch.Tensor,
        actual_success: torch.Tensor,
        predicted_confidence: torch.Tensor,
    ) -> Dict[str, float]:
        """Update metacognition based on prediction outcomes.

        Args:
            task_embedding: Task representation
            predicted_success: Model's prediction of success
            actual_success: Actual outcome (0 or 1)
            predicted_confidence: Model's confidence in prediction

        Returns:
            Training metrics
        """
        self.optimizer.zero_grad()

        # Add to calibration buffer
        self.calibrator.add_observation(predicted_confidence, actual_success)

        # Calibration loss
        calibration_loss = self.calibrator.get_calibration_loss()

        # Self-assessment loss: how accurate was the success prediction
        success_loss = F.binary_cross_entropy_with_logits(
            predicted_success.flatten(),
            actual_success.float().flatten()
        )

        # Total loss
        total_loss = (
            self.calibration_weight * calibration_loss +
            self.self_assessment_weight * success_loss
        )

        if total_loss.requires_grad:
            total_loss.backward()
            self.optimizer.step()

        # Track accuracy
        predicted = (predicted_success > 0).float()
        self.total_predictions += predicted.numel()
        self.correct_predictions += (predicted == actual_success.float()).sum().item()

        return {
            "calibration_loss": calibration_loss.item(),
            "success_loss": success_loss.item(),
            "total_loss": total_loss.item(),
            "prediction_accuracy": self.correct_predictions / max(self.total_predictions, 1),
            **self.calibrator.get_metrics(),
        }

    def should_trust_prediction(
        self,
        confidence: torch.Tensor,
        threshold: float = 0.7
    ) -> torch.Tensor:
        """Determine if prediction should be trusted based on calibrated confidence."""
        calibrated = self.calibrator.calibrate(confidence)
        return calibrated > threshold


def calibration_loss(
    confidence: torch.Tensor,
    outcomes: torch.Tensor,
    method: str = "brier"
) -> torch.Tensor:
    """Compute calibration loss for training.

    Args:
        confidence: Predicted confidences
        outcomes: Actual outcomes (0 or 1)
        method: "brier" for Brier score, "ece" for ECE

    Returns:
        Calibration loss value
    """
    if method == "ece":
        ece, _ = expected_calibration_error(confidence, outcomes)
        return ece
    else:
        return brier_score(confidence, outcomes)
