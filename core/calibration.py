"""Confidence calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


def brier_score(confidence: torch.Tensor, outcomes: torch.Tensor) -> torch.Tensor:
    if confidence.shape != outcomes.shape:
        raise ValueError("confidence and outcomes must have the same shape")
    return torch.mean((confidence - outcomes) ** 2)


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
