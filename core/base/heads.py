"""Output heads for vAGI."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn


class LanguageHead(nn.Module):
    """Project hidden states to vocabulary logits."""

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PolicyHead(nn.Module):
    """Project a pooled hidden state to action logits."""

    def __init__(self, hidden_size: int, action_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, action_dim)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        return self.proj(h_last)


class ActionValidityHead(nn.Module):
    """Predict action validity logits."""

    def __init__(self, hidden_size: int, action_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, action_dim)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        return self.proj(h_last)


class ValueHead(nn.Module):
    """Project a pooled hidden state to a scalar value."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        return self.proj(h_last)


class ConfidenceHead(nn.Module):
    """Predict a confidence value in [0, 1]."""

    def __init__(self, hidden_size: int, out_dim: int = 1) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(h_last))


class LogVarHead(nn.Module):
    """Predict a log-variance value for Gaussian uncertainty."""

    def __init__(self, hidden_size: int, out_dim: int = 1) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        return self.proj(h_last)


class WorldHead(nn.Module):
    """Project a pooled hidden state to multi-step observation prediction."""

    def __init__(self, hidden_size: int, obs_dim: int, horizon: int = 1) -> None:
        super().__init__()
        if horizon <= 0:
            raise ValueError("horizon must be > 0")
        self.obs_dim = obs_dim
        self.horizon = horizon
        self.proj = nn.Linear(hidden_size, obs_dim * horizon)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        out = self.proj(h_last)
        return out.view(h_last.shape[0], self.horizon, self.obs_dim)


class ErrorTypeHead(nn.Module):
    """Predict error-type logits for reflection."""

    def __init__(self, hidden_size: int, error_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, error_dim)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        return self.proj(h_last)


class InfoGainHead(nn.Module):
    """Predict information-gain score for active learning (Issue 2.8).

    This head predicts how much information would be gained by taking different
    actions or asking clarifying questions. Higher scores indicate states where
    the model would benefit most from new information.

    Usage:
        1. Action Selection: Use info_gain as exploration bonus
           - action_score = policy_logits + info_gain_weight * info_gain
        2. Active Learning: Query human when info_gain exceeds threshold
           - if info_gain > query_threshold: request_human_feedback()
        3. Curriculum Learning: Prioritize training on high info_gain examples
    """

    def __init__(self, hidden_size: int, query_threshold: float = 0.7) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        self.query_threshold = query_threshold
        # Optional: per-action info gain for more granular exploration
        self.per_action_proj = None  # Can be set via set_action_dim()

    def set_action_dim(self, action_dim: int) -> None:
        """Enable per-action information gain prediction."""
        device = self.proj.weight.device
        self.per_action_proj = nn.Linear(self.proj.in_features, action_dim).to(device)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        """Predict overall information gain score in [0, 1]."""
        return torch.sigmoid(self.proj(h_last))

    def forward_per_action(self, h_last: torch.Tensor) -> torch.Tensor:
        """Predict per-action information gain scores.

        Returns:
            Per-action info gain (B, action_dim) in [0, 1], or None if not configured
        """
        if self.per_action_proj is None:
            return None
        return torch.sigmoid(self.per_action_proj(h_last))

    def should_query(self, info_gain: torch.Tensor) -> torch.Tensor:
        """Determine if human feedback should be requested (active learning).

        Args:
            info_gain: Information gain scores (B, 1)

        Returns:
            Boolean tensor (B,) indicating whether to query for each sample
        """
        return (info_gain.squeeze(-1) > self.query_threshold)

    def exploration_bonus(
        self,
        action_logits: torch.Tensor,
        info_gain: torch.Tensor,
        weight: float = 1.0,
    ) -> torch.Tensor:
        """Add information-gain based exploration bonus to action logits.

        Args:
            action_logits: Raw policy logits (B, action_dim)
            info_gain: Information gain score (B, 1)
            weight: Bonus weight (higher = more exploration)

        Returns:
            Modified action logits with exploration bonus
        """
        # Scale bonus by info_gain - explore more when uncertain
        bonus = weight * info_gain
        return action_logits + bonus


class BudgetHead(nn.Module):
    """Predict compute budget decisions for adaptive planning (Issue 2.9).

    This head enables the model to dynamically allocate compute resources based
    on task complexity. It predicts:
    - mode: Whether to act immediately (0) or think/plan first (1)
    - horizon: How many steps to look ahead during planning
    - candidates: How many action candidates to evaluate

    Usage Examples:
        1. Basic usage in planning:
            mode_logits, horizon_logits, candidate_logits = budget_head(h_policy)
            budget = budget_head.decode_budget(mode_logits, horizon_logits, candidate_logits)
            if budget["mode"] == "act":
                action = model.act(...)
            else:
                action = model.plan_step(..., horizon=budget["horizon"],
                                         num_candidates=budget["num_candidates"])

        2. With uncertainty-based adjustment:
            budget = budget_head.decode_budget(...)
            if uncertainty > threshold:
                budget["horizon"] = min(budget["horizon"] * 2, max_horizon)

        3. Training target generation:
            # For simple states: mode=0 (act), horizon=1, candidates=1
            # For complex states: mode=1 (think), horizon=4+, candidates=8+
            targets = {
                "budget_mode": torch.tensor([1]),  # think
                "budget_horizon": torch.tensor([3]),  # 4 steps (0-indexed)
                "budget_candidates": torch.tensor([7]),  # 8 candidates (0-indexed)
            }

    Integration with Planning:
        The think_then_act() method in VAGICore automatically uses BudgetHead
        predictions to determine planning depth. Higher uncertainty or complexity
        leads to more thorough planning.
    """

    def __init__(self, hidden_size: int, max_horizon: int, max_candidates: int) -> None:
        super().__init__()
        self.max_horizon = max_horizon
        self.max_candidates = max_candidates
        self.mode = nn.Linear(hidden_size, 2)  # 0=act, 1=think
        self.horizon = nn.Linear(hidden_size, max_horizon)
        self.candidates = nn.Linear(hidden_size, max_candidates)

    def forward(self, h_last: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict budget allocation logits.

        Returns:
            mode_logits: (B, 2) - act vs think
            horizon_logits: (B, max_horizon) - planning depth
            candidate_logits: (B, max_candidates) - search breadth
        """
        return self.mode(h_last), self.horizon(h_last), self.candidates(h_last)

    def decode_budget(
        self,
        mode_logits: torch.Tensor,
        horizon_logits: torch.Tensor,
        candidate_logits: torch.Tensor,
    ) -> Dict[str, Any]:
        """Decode budget logits into planning parameters.

        Args:
            mode_logits: Mode prediction logits (B, 2)
            horizon_logits: Horizon prediction logits (B, max_horizon)
            candidate_logits: Candidate prediction logits (B, max_candidates)

        Returns:
            Dictionary with mode ("act" or "think"), horizon, and num_candidates
        """
        mode = "think" if int(torch.argmax(mode_logits, dim=-1)[0].item()) == 1 else "act"
        horizon = int(torch.argmax(horizon_logits, dim=-1)[0].item()) + 1
        candidates = int(torch.argmax(candidate_logits, dim=-1)[0].item()) + 1
        horizon = max(1, min(self.max_horizon, horizon))
        candidates = max(1, min(self.max_candidates, candidates))
        return {"mode": mode, "horizon": horizon, "num_candidates": candidates}

    def adjust_for_uncertainty(
        self,
        budget: Dict[str, Any],
        uncertainty: float,
        uncertainty_threshold: float = 0.5,
        max_horizon_boost: int = 2,
    ) -> Dict[str, Any]:
        """Adjust budget based on uncertainty level (more compute for uncertain states).

        Args:
            budget: Decoded budget dictionary
            uncertainty: Current uncertainty estimate
            uncertainty_threshold: Threshold above which to increase compute
            max_horizon_boost: Maximum multiplier for horizon

        Returns:
            Adjusted budget dictionary
        """
        if uncertainty > uncertainty_threshold:
            boost = min(max_horizon_boost, 1 + int(uncertainty / uncertainty_threshold))
            budget["horizon"] = min(budget["horizon"] * boost, self.max_horizon)
            budget["num_candidates"] = min(budget["num_candidates"] * boost, self.max_candidates)
            if budget["mode"] == "act":
                budget["mode"] = "think"  # Switch to thinking under uncertainty
        return budget
