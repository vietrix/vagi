"""Model adapter for vAGI serving."""

from __future__ import annotations

from typing import Optional

import torch

from vagi_core import RecurrentState, VAGIConfig, VAGICore


class CoreAdapter:
    def __init__(self, cfg: VAGIConfig, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = VAGICore(cfg).to(self.device)
        self.model.eval()

    def init_state(self, batch_size: int) -> RecurrentState:
        return self.model.init_state(batch_size=batch_size, device=self.device)

    def step(
        self,
        *,
        input_ids: torch.Tensor,
        obs: Optional[torch.Tensor],
        state: RecurrentState,
        task_ids: Optional[torch.Tensor] = None,
    ) -> dict:
        return self.model.step(input_ids=input_ids, obs=obs, state=state, task_ids=task_ids)

    def plan(
        self,
        *,
        input_ids: torch.Tensor,
        obs: Optional[torch.Tensor],
        state: RecurrentState,
        task_ids: Optional[torch.Tensor],
        num_candidates: int,
        horizon: int,
        uncertainty_weight: float,
        info_gain_weight: float,
        strategy: str,
    ) -> dict:
        return self.model.plan_step(
            input_ids=input_ids,
            obs=obs,
            state=state,
            task_ids=task_ids,
            num_candidates=num_candidates,
            horizon=horizon,
            uncertainty_weight=uncertainty_weight,
            info_gain_weight=info_gain_weight,
            strategy=strategy,
        )
