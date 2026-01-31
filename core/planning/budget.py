"""Budget controller utilities for compute-aware planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from ..training.calibration import ConfidenceCalibrator


@dataclass
class BudgetDecision:
    mode: str
    horizon: int
    num_candidates: int
    reason: str
    confidence: float


@dataclass
class CounterfactualRecord:
    uncertainty: float
    value_spread: float
    task_difficulty: float
    delta_reward: float
    delta_latency: float


class BudgetController:
    """Auto-tune planning budget to minimize compute while preserving reward."""

    def __init__(
        self,
        *,
        max_horizon: int = 4,
        max_candidates: int = 8,
        min_confidence_to_act: float = 0.0,
        compute_weight: float = 0.0,
        weights: Optional[Iterable[float]] = None,
        bias: float = 0.0,
        calibrator: Optional[ConfidenceCalibrator] = None,
    ) -> None:
        if max_horizon <= 0:
            raise ValueError("max_horizon must be > 0")
        if max_candidates <= 0:
            raise ValueError("max_candidates must be > 0")
        self.max_horizon = int(max_horizon)
        self.max_candidates = int(max_candidates)
        self.min_confidence_to_act = float(min_confidence_to_act)
        self.compute_weight = float(compute_weight)
        if weights is None:
            weights = [1.0, 1.0, 1.0]
        self.weights = torch.tensor(list(weights), dtype=torch.float32)
        self.bias = float(bias)
        self.calibrator = calibrator

    def _normalize_features(
        self,
        uncertainty: float,
        value_spread: float,
        task_difficulty: float,
    ) -> torch.Tensor:
        u = float(uncertainty)
        v = float(value_spread)
        d = float(task_difficulty)
        u_norm = u / (1.0 + u)
        v_norm = v / (1.0 + v)
        d_norm = max(0.0, min(1.0, d))
        return torch.tensor([u_norm, v_norm, d_norm], dtype=torch.float32)

    def _plan_score(self, features: torch.Tensor) -> float:
        logits = torch.dot(features, self.weights) + self.bias
        return float(torch.sigmoid(logits).item())

    def decide(
        self,
        *,
        uncertainty: float,
        value_spread: float,
        task_difficulty: float = 0.5,
        policy_only: bool = False,
    ) -> BudgetDecision:
        features = self._normalize_features(uncertainty, value_spread, task_difficulty)
        confidence = 1.0 / (1.0 + float(uncertainty))
        if self.calibrator is not None:
            conf_tensor = torch.tensor([confidence], dtype=torch.float32)
            confidence = float(self.calibrator.apply(conf_tensor)[0].item())

        if policy_only:
            return BudgetDecision(
                mode="act",
                horizon=1,
                num_candidates=1,
                reason="policyOnly",
                confidence=confidence,
            )

        if confidence < self.min_confidence_to_act:
            return BudgetDecision(
                mode="act",
                horizon=1,
                num_candidates=1,
                reason="needsInfo",
                confidence=confidence,
            )

        plan_prob = self._plan_score(features)
        if plan_prob < 0.5:
            return BudgetDecision(
                mode="act",
                horizon=1,
                num_candidates=1,
                reason="lowGain",
                confidence=confidence,
            )

        compute_scale = float(features.mean().item())
        horizon = 1 + int(round((self.max_horizon - 1) * compute_scale))
        candidates = 1 + int(round((self.max_candidates - 1) * compute_scale))
        horizon = max(1, min(self.max_horizon, horizon))
        candidates = max(1, min(self.max_candidates, candidates))
        return BudgetDecision(
            mode="think",
            horizon=horizon,
            num_candidates=candidates,
            reason="plan",
            confidence=confidence,
        )

    def update_from_counterfactuals(
        self,
        records: Iterable[CounterfactualRecord],
        *,
        reward_margin: float = 0.0,
        steps: int = 200,
        lr: float = 0.1,
    ) -> None:
        features = []
        labels = []
        for record in records:
            feat = self._normalize_features(record.uncertainty, record.value_spread, record.task_difficulty)
            utility = record.delta_reward - self.compute_weight * record.delta_latency
            label = 1.0 if utility > reward_margin else 0.0
            features.append(feat)
            labels.append(label)
        if not features:
            return
        feats = torch.stack(features)
        targets = torch.tensor(labels, dtype=torch.float32)
        weights = torch.nn.Parameter(self.weights.clone())
        bias = torch.nn.Parameter(torch.tensor(self.bias))
        optimizer = torch.optim.Adam([weights, bias], lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        for _ in range(steps):
            optimizer.zero_grad()
            logits = feats @ weights + bias
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

        self.weights = weights.detach()
        self.bias = float(bias.detach().item())

    def calibrate_confidence(
        self, confidence: torch.Tensor, outcomes: torch.Tensor, *, steps: int = 200, lr: float = 0.1
    ) -> ConfidenceCalibrator:
        calibrator = self.calibrator or ConfidenceCalibrator()
        calibrator.fit(confidence, outcomes, steps=steps, lr=lr)
        self.calibrator = calibrator
        return calibrator

    def to_dict(self) -> dict:
        return {
            "max_horizon": self.max_horizon,
            "max_candidates": self.max_candidates,
            "min_confidence_to_act": self.min_confidence_to_act,
            "compute_weight": self.compute_weight,
            "weights": [float(w) for w in self.weights.tolist()],
            "bias": self.bias,
            "calibration": {
                "temperature": self.calibrator.temperature if self.calibrator else None,
                "bias": self.calibrator.bias if self.calibrator else None,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BudgetController":
        calib = payload.get("calibration") or {}
        calibrator = None
        if calib.get("temperature") is not None:
            calibrator = ConfidenceCalibrator(
                temperature=float(calib.get("temperature", 1.0)),
                bias=float(calib.get("bias", 0.0)),
            )
        return cls(
            max_horizon=int(payload.get("max_horizon", 4)),
            max_candidates=int(payload.get("max_candidates", 8)),
            min_confidence_to_act=float(payload.get("min_confidence_to_act", 0.0)),
            compute_weight=float(payload.get("compute_weight", 0.0)),
            weights=payload.get("weights", [1.0, 1.0, 1.0]),
            bias=float(payload.get("bias", 0.0)),
            calibrator=calibrator,
        )
