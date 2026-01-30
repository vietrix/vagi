"""Main vAGI model."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from .backbone import CausalTransformerBackbone
from .config import VAGIConfig
from .heads import (
    BudgetHead,
    ErrorTypeHead,
    InfoGainHead,
    LanguageHead,
    LogVarHead,
    PolicyHead,
    ValueHead,
    WorldHead,
)
from .losses import (
    budget_loss,
    imagination_consistency_loss,
    language_loss,
    policy_loss,
    reflection_loss,
    total_loss,
    value_loss,
    world_loss,
)
from .memory import KVCache, RecurrentState
from .utils import check_floating, check_shape, sanitize_tensor, validate_seq_len, StageTimer
from .vision import ImageObsEncoder


class VAGICore(nn.Module):
    """vAGI causal transformer with recurrent state."""

    def __init__(self, cfg: VAGIConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = CausalTransformerBackbone(cfg)
        self.lang = LanguageHead(cfg.hidden_size, cfg.vocab_size)
        self.pi = PolicyHead(cfg.hidden_size, cfg.action_dim)
        self.v = ValueHead(cfg.hidden_size)
        self.value_logvar = LogVarHead(cfg.hidden_size, 1) if cfg.use_uncertainty else None
        self.error_head = ErrorTypeHead(cfg.hidden_size, cfg.error_type_dim) if cfg.use_reflection else None
        self.info_head = InfoGainHead(cfg.hidden_size) if cfg.use_reflection else None
        self.budget_head = (
            BudgetHead(cfg.hidden_size, cfg.budget_max_horizon, cfg.budget_max_candidates)
            if cfg.use_budget_head
            else None
        )
        self.world = (
            WorldHead(cfg.hidden_size, cfg.obs_dim, horizon=cfg.world_model_horizon)
            if cfg.use_world_pred
            else None
        )
        self.world_logvar = (
            LogVarHead(cfg.hidden_size, cfg.obs_dim * cfg.world_model_horizon)
            if cfg.use_uncertainty and cfg.use_world_pred
            else None
        )
        self.vision = (
            ImageObsEncoder(cfg.vision_channels, cfg.obs_dim, hidden_size=cfg.vision_hidden)
            if cfg.use_vision
            else None
        )

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device | str] = None,
        *,
        prefill_kv: bool = False,
        kv_max_seq_len: Optional[int] = None,
    ) -> RecurrentState:
        """Initialize a zeroed recurrent state with optional KV preallocation."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        device = device or next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        mem = torch.zeros(
            (batch_size, self.cfg.memory_slots, self.cfg.hidden_size),
            device=device,
            dtype=dtype,
        )
        if kv_max_seq_len is not None and kv_max_seq_len <= 0:
            raise ValueError("kv_max_seq_len must be > 0")
        if prefill_kv:
            max_len = kv_max_seq_len or self.cfg.max_seq_len
            n_kv_heads = self.cfg.n_kv_heads if self.cfg.use_gqa else self.cfg.n_heads
            kv = KVCache.allocate(
                num_layers=self.cfg.n_layers,
                batch_size=batch_size,
                n_kv_heads=n_kv_heads,
                head_dim=self.cfg.head_dim,
                max_len=max_len,
                device=device,
                dtype=dtype,
            )
        else:
            kv = KVCache.empty(self.cfg.n_layers)
            if kv_max_seq_len is not None:
                kv.max_len = kv_max_seq_len
        return RecurrentState(mem=mem, kv=kv, timestep=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        obs: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
        state: Optional[RecurrentState] = None,
        labels: Optional[torch.Tensor] = None,
        targets: Optional[Dict[str, Any]] = None,
        return_loss: bool = False,
        return_hidden: bool = False,
        image: Optional[torch.Tensor] = None,
        timer: Optional[StageTimer] = None,
    ) -> Dict[str, Any]:
        """Forward pass for full sequences."""
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must be torch.long")
        check_shape(input_ids, (None, None), "input_ids")
        validate_seq_len(input_ids, self.cfg.max_seq_len, name="input_ids")

        obs = self._resolve_obs(obs, image)
        obs = self._sanitize_obs(obs)
        if obs is not None:
            check_shape(obs, (input_ids.shape[0], self.cfg.obs_dim), "obs")
            check_floating(obs, "obs")

        if state is not None:
            if not isinstance(state, RecurrentState):
                raise TypeError("state must be a RecurrentState")
            check_shape(state.mem, (input_ids.shape[0], self.cfg.memory_slots, self.cfg.hidden_size), "state.mem")

        x, h_last, h_act, mem_next = self.backbone(
            input_ids=input_ids,
            obs=obs,
            task_ids=task_ids,
            state=state,
            timer=timer,
        )

        text_logits = self.lang(x)
        h_policy = h_act if h_act is not None else h_last
        action_logits = self.pi(h_policy)
        if timer:
            with timer.track("value"):
                value = self.v(h_policy)
        else:
            value = self.v(h_policy)
        if timer and self.world is not None:
            with timer.track("world"):
                world_pred = self.world(h_policy)
        else:
            world_pred = self.world(h_policy) if self.world is not None else None
        value_logvar = self.value_logvar(h_policy) if self.value_logvar is not None else None
        world_logvar = self._format_world_logvar(h_policy) if self.world_logvar is not None else None
        value_logvar = self._apply_obs_uncertainty(value_logvar, obs)
        world_logvar = self._apply_obs_uncertainty(world_logvar, obs)
        error_logits = self.error_head(h_policy) if self.error_head is not None else None
        info_gain = self.info_head(h_policy) if self.info_head is not None else None
        budget_mode = None
        budget_horizon = None
        budget_candidates = None
        if self.budget_head is not None:
            budget_mode, budget_horizon, budget_candidates = self.budget_head(h_policy)
        uncertainty = self._uncertainty_tensor(
            world_logvar,
            value_logvar,
            batch_size=input_ids.shape[0],
            device=input_ids.device,
        )
        confidence = self._confidence_from_uncertainty(uncertainty)
        budget_summary = self._budget_summary(budget_mode, budget_horizon, budget_candidates)

        state_out = None
        if state is not None:
            state_out = RecurrentState(
                mem=mem_next if mem_next is not None else state.mem,
                kv=state.kv,
                timestep=state.timestep,
            )

        outputs: Dict[str, Any] = {
            "text_logits": text_logits,
            "action_logits": action_logits,
            "value": value,
            "value_logvar": value_logvar,
            "world_pred": world_pred,
            "world_logvar": world_logvar,
            "error_logits": error_logits,
            "info_gain": info_gain,
            "budget_mode_logits": budget_mode,
            "budget_horizon_logits": budget_horizon,
            "budget_candidate_logits": budget_candidates,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "budget": budget_summary,
            "stopReason": None,
            "state": state_out,
        }
        if return_hidden:
            outputs["hidden"] = h_last

        if return_loss:
            losses: Dict[str, torch.Tensor] = {}
            if labels is not None:
                if labels.dtype != torch.long:
                    raise TypeError("labels must be torch.long")
                include_special = self.cfg.use_special_tokens
                k_prefix = self.cfg.obs_tokens if obs is not None else 0
                k_suffix = 1 if include_special else 0
                losses["language"] = language_loss(text_logits, labels, k_prefix=k_prefix, k_suffix=k_suffix)
            targets = targets or {}
            if "actions" in targets:
                losses["policy"] = policy_loss(action_logits, targets["actions"])
            if "values" in targets:
                losses["value"] = value_loss(value, targets["values"], logvar=value_logvar)
            if world_pred is not None:
                if "obs_future" in targets:
                    losses["world"] = world_loss(world_pred, targets["obs_future"], logvar=world_logvar)
                elif "obs_next" in targets:
                    losses["world"] = world_loss(world_pred, targets["obs_next"], logvar=world_logvar)
                if targets.get("world_consistency") or "world_consistency" in (targets.get("loss_weights") or {}):
                    max_delta = float(targets.get("world_consistency_max_delta", 1.0))
                    losses["world_consistency"] = imagination_consistency_loss(
                        world_pred,
                        world_logvar,
                        max_delta=max_delta,
                    )
            reflection = reflection_loss(
                error_logits,
                targets.get("error_types") if targets else None,
                info_gain,
                targets.get("info_gain") if targets else None,
            )
            losses.update(reflection)
            losses.update(
                budget_loss(
                    budget_mode,
                    budget_horizon,
                    budget_candidates,
                    targets,
                )
            )

            if losses:
                outputs["loss"] = total_loss(losses, weights=targets.get("loss_weights"))
                outputs["losses_breakdown"] = losses
            else:
                outputs["loss"] = None
                outputs["losses_breakdown"] = {}

        return outputs

    @torch.no_grad()
    def step(
        self,
        input_ids: torch.Tensor,
        obs: Optional[torch.Tensor],
        state: RecurrentState,
        task_ids: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        timer: Optional[StageTimer] = None,
    ) -> Dict[str, Any]:
        """Single-step inference with recurrent state update."""
        if state is None:
            raise ValueError("state is required for step()")
        validate_seq_len(input_ids, self.cfg.max_seq_len, name="input_ids")
        obs = self._resolve_obs(obs, image)
        obs = self._sanitize_obs(obs)
        if obs is None:
            raise ValueError("obs is required for step()")
        check_shape(obs, (input_ids.shape[0], self.cfg.obs_dim), "obs")
        check_floating(obs, "obs")
        x, h_last, h_act, mem_next, kv_next = self.backbone.forward_step(
            input_ids=input_ids,
            obs=obs,
            task_ids=task_ids,
            state=state,
            timer=timer,
        )
        h_policy = h_act if h_act is not None else h_last
        if timer:
            with timer.track("value"):
                value = self.v(h_policy)
            if self.world is not None:
                with timer.track("world"):
                    world_pred = self.world(h_policy)
            else:
                world_pred = None
        else:
            value = self.v(h_policy)
            world_pred = self.world(h_policy) if self.world is not None else None

        outputs: Dict[str, Any] = {
            "text_logits": self.lang(x),
            "action_logits": self.pi(h_policy),
            "value": value,
            "value_logvar": None,
            "world_pred": world_pred,
            "world_logvar": None,
            "error_logits": self.error_head(h_policy) if self.error_head is not None else None,
            "info_gain": self.info_head(h_policy) if self.info_head is not None else None,
            "budget_mode_logits": None,
            "budget_horizon_logits": None,
            "budget_candidate_logits": None,
            "state": None,
        }
        if self.budget_head is not None:
            mode_logits, horizon_logits, candidate_logits = self.budget_head(h_policy)
            outputs["budget_mode_logits"] = mode_logits
            outputs["budget_horizon_logits"] = horizon_logits
            outputs["budget_candidate_logits"] = candidate_logits
        budget_summary = self._budget_summary(
            outputs.get("budget_mode_logits"),
            outputs.get("budget_horizon_logits"),
            outputs.get("budget_candidate_logits"),
        )
        if self.value_logvar is not None:
            outputs["value_logvar"] = self._apply_obs_uncertainty(self.value_logvar(h_policy), obs)
        if self.world_logvar is not None:
            outputs["world_logvar"] = self._apply_obs_uncertainty(self._format_world_logvar(h_policy), obs)
        uncertainty = self._uncertainty_tensor(
            outputs.get("world_logvar"),
            outputs.get("value_logvar"),
            batch_size=input_ids.shape[0],
            device=input_ids.device,
        )
        outputs["confidence"] = self._confidence_from_uncertainty(uncertainty)
        outputs["uncertainty"] = uncertainty
        outputs["budget"] = budget_summary
        outputs["stopReason"] = None
        if outputs["state"] is not None:
            state_out = outputs["state"]
            outputs["state"] = RecurrentState(
                mem=state_out.mem,
                kv=state_out.kv,
                timestep=state_out.timestep + 1,
            )
        else:
            outputs["state"] = RecurrentState(
                mem=mem_next if mem_next is not None else state.mem,
                kv=kv_next,
                timestep=state.timestep + 1,
            )
        return outputs

    @torch.no_grad()
    def act(
        self,
        input_ids: torch.Tensor,
        obs: Optional[torch.Tensor],
        state: RecurrentState,
        task_ids: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Fast policy-only action selection."""
        out = self.step(input_ids=input_ids, obs=obs, state=state, task_ids=task_ids, image=image)
        action = torch.argmax(out["action_logits"], dim=-1)
        return {
            "action": action,
            "mode": "act",
            "confidence": out.get("confidence"),
            "uncertainty": out.get("uncertainty"),
            "budget": out.get("budget"),
            "stopReason": "done",
            "outputs": out,
        }

    @torch.no_grad()
    def think_then_act(
        self,
        input_ids: torch.Tensor,
        obs: Optional[torch.Tensor],
        state: RecurrentState,
        task_ids: Optional[torch.Tensor] = None,
        *,
        horizon: Optional[int] = None,
        num_candidates: Optional[int] = None,
        strategy: str = "cem",
        uncertainty_weight: float = 1.0,
        info_gain_weight: float = 0.0,
        max_horizon: Optional[int] = None,
        max_candidates: Optional[int] = None,
        max_steps: Optional[int] = None,
        risk_penalty: Optional[float] = None,
        min_confidence_to_act: Optional[float] = None,
        policy_only: bool = False,
        trace: bool = False,
        uncertainty_fallback: Optional[float] = None,
        error_stop_prob: Optional[float] = None,
        error_stop_ids: Optional[list[int]] = None,
        image: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Plan then act using reflection + info gain with budget control."""
        if policy_only:
            return self.act(input_ids=input_ids, obs=obs, state=state, task_ids=task_ids, image=image)
        budget = None
        if self.budget_head is not None:
            base = self.forward(
                input_ids=input_ids,
                obs=obs,
                task_ids=task_ids,
                state=state,
                return_loss=False,
                image=image,
            )
            budget = self._budget_from_logits(
                base.get("budget_mode_logits"),
                base.get("budget_horizon_logits"),
                base.get("budget_candidate_logits"),
            )
            if budget["mode"] == "act":
                act_out = self.act(input_ids=input_ids, obs=obs, state=state, task_ids=task_ids, image=image)
                act_out["budget"] = budget
                return act_out
            horizon = budget["horizon"]
            num_candidates = budget["num_candidates"]

        if self.world is None:
            return self.act(input_ids=input_ids, obs=obs, state=state, task_ids=task_ids, image=image)

        plan = self.plan_step(
            input_ids=input_ids,
            obs=obs,
            state=state,
            task_ids=task_ids,
            num_candidates=num_candidates or 4,
            horizon=horizon or 3,
            uncertainty_weight=uncertainty_weight,
            info_gain_weight=info_gain_weight,
            max_horizon=max_horizon,
            max_candidates=max_candidates,
            max_steps=max_steps,
            risk_penalty=risk_penalty,
            min_confidence_to_act=min_confidence_to_act,
            policy_only=policy_only,
            trace=trace,
            strategy=strategy,
            uncertainty_fallback=uncertainty_fallback,
            error_stop_prob=error_stop_prob,
            error_stop_ids=error_stop_ids,
            image=image,
        )
        plan["budget"] = budget
        return plan

    def _budget_from_logits(
        self,
        mode_logits: Optional[torch.Tensor],
        horizon_logits: Optional[torch.Tensor],
        candidate_logits: Optional[torch.Tensor],
    ) -> Dict[str, int | str]:
        if mode_logits is None or horizon_logits is None or candidate_logits is None:
            return {"mode": "think", "horizon": self.cfg.budget_max_horizon, "num_candidates": self.cfg.budget_max_candidates}
        mode = "think" if int(torch.argmax(mode_logits, dim=-1)[0].item()) == 1 else "act"
        horizon = int(torch.argmax(horizon_logits, dim=-1)[0].item()) + 1
        candidates = int(torch.argmax(candidate_logits, dim=-1)[0].item()) + 1
        horizon = max(1, min(self.cfg.budget_max_horizon, horizon))
        candidates = max(1, min(self.cfg.budget_max_candidates, candidates))
        return {"mode": mode, "horizon": horizon, "num_candidates": candidates}

    @torch.no_grad()
    def plan_step(
        self,
        input_ids: torch.Tensor,
        obs: Optional[torch.Tensor],
        state: RecurrentState,
        task_ids: Optional[torch.Tensor] = None,
        num_candidates: int = 4,
        horizon: int = 3,
        uncertainty_weight: float = 1.0,
        info_gain_weight: float = 0.0,
        max_horizon: Optional[int] = None,
        max_candidates: Optional[int] = None,
        max_steps: Optional[int] = None,
        risk_penalty: Optional[float] = None,
        min_confidence_to_act: Optional[float] = None,
        policy_only: bool = False,
        trace: bool = False,
        strategy: str = "cem",
        cem_iters: int = 3,
        elite_frac: float = 0.2,
        tree_branching: int = 4,
        uncertainty_fallback: Optional[float] = None,
        error_stop_prob: Optional[float] = None,
        error_stop_ids: Optional[list[int]] = None,
        image: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Plan a single action by rolling out the world model and scoring with value."""
        if num_candidates <= 0:
            raise ValueError("num_candidates must be > 0")
        if horizon <= 0:
            raise ValueError("horizon must be > 0")
        if cem_iters <= 0:
            raise ValueError("cem_iters must be > 0")
        if not (0.0 < elite_frac <= 1.0):
            raise ValueError("elite_frac must be in (0, 1]")
        if tree_branching <= 0:
            raise ValueError("tree_branching must be > 0")
        if self.world is None:
            raise ValueError("world model is required for plan_step")
        if state is None:
            raise ValueError("state is required for plan_step")
        validate_seq_len(input_ids, self.cfg.max_seq_len, name="input_ids")
        obs = self._resolve_obs(obs, image)
        obs = self._sanitize_obs(obs)
        if obs is None:
            raise ValueError("obs is required for plan_step")
        check_shape(obs, (input_ids.shape[0], self.cfg.obs_dim), "obs")
        check_floating(obs, "obs")

        base = self.forward(
            input_ids=input_ids,
            obs=obs,
            task_ids=task_ids,
            state=state,
            return_loss=False,
            image=image,
        )
        action_logits = base["action_logits"]
        base_uncertainty = self._mean_uncertainty(base.get("world_logvar"), base.get("value_logvar"))
        uncertainty_tensor = self._uncertainty_tensor(
            base.get("world_logvar"),
            base.get("value_logvar"),
            batch_size=input_ids.shape[0],
            device=input_ids.device,
        )
        confidence = self._confidence_from_uncertainty(uncertainty_tensor)
        budget_summary = self._budget_summary(
            base.get("budget_mode_logits"),
            base.get("budget_horizon_logits"),
            base.get("budget_candidate_logits"),
        )

        if risk_penalty is not None:
            uncertainty_weight = risk_penalty
        if max_horizon is not None:
            horizon = min(horizon, max_horizon)
            budget_summary["horizon"] = min(int(budget_summary.get("horizon", horizon)), horizon)
        if max_candidates is not None:
            num_candidates = min(num_candidates, max_candidates)
            budget_summary["num_candidates"] = min(
                int(budget_summary.get("num_candidates", num_candidates)), num_candidates
            )
        if max_steps is not None:
            horizon = min(horizon, max_steps)
            budget_summary["horizon"] = min(int(budget_summary.get("horizon", horizon)), horizon)

        stop_reason = None
        if policy_only:
            greedy = torch.argmax(action_logits, dim=-1)
            return {
                "action": greedy,
                "action_logits": action_logits,
                "candidate_actions": greedy.unsqueeze(-1),
                "candidate_values": torch.full(
                    (obs.shape[0], 1),
                    float(base["value"].mean().item()),
                    device=obs.device,
                    dtype=obs.dtype,
                ),
                "early_stop": True,
                "mode": "act",
                "confidence": confidence,
                "uncertainty": uncertainty_tensor,
                "budget": budget_summary,
                "stopReason": "done",
                "trace": None,
            }
        if error_stop_prob is not None and base.get("error_logits") is not None:
            probs = torch.softmax(base["error_logits"], dim=-1)
            max_prob, max_idx = torch.max(probs, dim=-1)
            if torch.any(max_prob >= error_stop_prob):
                if error_stop_ids is None or int(max_idx[0].item()) in error_stop_ids:
                    greedy = torch.argmax(action_logits, dim=-1)
                    return {
                        "action": greedy,
                        "action_logits": action_logits,
                        "candidate_actions": greedy.unsqueeze(-1),
                        "candidate_values": torch.full(
                            (obs.shape[0], 1),
                            float(base["value"].mean().item()),
                            device=obs.device,
                            dtype=obs.dtype,
                        ),
                        "early_stop": True,
                        "mode": "act",
                        "confidence": confidence,
                        "uncertainty": uncertainty_tensor,
                        "budget": budget_summary,
                        "stopReason": "done",
                        "trace": None,
                    }
        if uncertainty_fallback is not None:
            base_uncertainty = self._mean_uncertainty(base.get("world_logvar"), base.get("value_logvar"))
            if base_uncertainty > uncertainty_fallback:
                greedy = torch.argmax(action_logits, dim=-1)
                return {
                    "action": greedy,
                    "action_logits": action_logits,
                    "candidate_actions": greedy.unsqueeze(-1),
                    "candidate_values": torch.full(
                        (obs.shape[0], 1),
                        float(base["value"].mean().item()),
                        device=obs.device,
                        dtype=obs.dtype,
                    ),
                    "early_stop": False,
                    "mode": "act",
                    "confidence": confidence,
                    "uncertainty": uncertainty_tensor,
                    "budget": budget_summary,
                    "stopReason": "unsure",
                    "trace": None,
                }
        if min_confidence_to_act is not None and float(confidence.mean().item()) < min_confidence_to_act:
            greedy = torch.argmax(action_logits, dim=-1)
            return {
                "action": greedy,
                "action_logits": action_logits,
                "candidate_actions": greedy.unsqueeze(-1),
                "candidate_values": torch.full(
                    (obs.shape[0], 1),
                    float(base["value"].mean().item()),
                    device=obs.device,
                    dtype=obs.dtype,
                ),
                "early_stop": False,
                "mode": "act",
                "confidence": confidence,
                "uncertainty": uncertainty_tensor,
                "budget": budget_summary,
                "stopReason": "needsInfo",
                "trace": None,
            }

        if base.get("value_logvar") is not None and horizon > 1:
            value_var = float(torch.exp(base["value_logvar"]).mean().item())
            if value_var > 1.0:
                horizon = max(1, horizon // 2)
                uncertainty_weight = max(uncertainty_weight, uncertainty_weight * 1.5)
                stop_reason = "unsure"

        strategy_key = strategy.lower().strip()
        if strategy_key == "sample":
            result = self._plan_sample(
                action_logits=action_logits,
                obs=obs,
                task_ids=task_ids,
                state=state,
                num_candidates=num_candidates,
                horizon=horizon,
                uncertainty_weight=uncertainty_weight,
                info_gain_weight=info_gain_weight,
                trace=trace,
            )
            result["mode"] = "think"
            result["confidence"] = confidence
            result["uncertainty"] = uncertainty_tensor
            result["budget"] = budget_summary
            result["stopReason"] = stop_reason
            return result
        if strategy_key == "tree":
            result = self._plan_tree(
                action_logits=action_logits,
                obs=obs,
                task_ids=task_ids,
                state=state,
                horizon=horizon,
                uncertainty_weight=uncertainty_weight,
                info_gain_weight=info_gain_weight,
                branch=tree_branching,
                trace=trace,
            )
            result["mode"] = "think"
            result["confidence"] = confidence
            result["uncertainty"] = uncertainty_tensor
            result["budget"] = budget_summary
            result["stopReason"] = stop_reason
            return result
        result = self._plan_cem(
            action_logits=action_logits,
            obs=obs,
            task_ids=task_ids,
            state=state,
            num_candidates=num_candidates,
            horizon=horizon,
            uncertainty_weight=uncertainty_weight,
            info_gain_weight=info_gain_weight,
            cem_iters=cem_iters,
            elite_frac=elite_frac,
            trace=trace,
        )
        result["mode"] = "think"
        result["confidence"] = confidence
        result["uncertainty"] = uncertainty_tensor
        result["budget"] = budget_summary
        result["stopReason"] = stop_reason
        return result

    def _score_sequence(
        self,
        obs: torch.Tensor,
        state: RecurrentState,
        sequence: torch.Tensor,
        *,
        task_ids: Optional[torch.Tensor],
        uncertainty_weight: float,
        info_gain_weight: float,
    ) -> float:
        obs_roll = obs
        state_roll = state
        value = None
        uncertainty = 0.0
        uncertainty_steps = 0
        info_gain_score = 0.0
        info_gain_steps = 0
        for action in sequence:
            action_tensor = action.view(1, 1)
            out = self.step(input_ids=action_tensor, obs=obs_roll, state=state_roll, task_ids=task_ids)
            world_pred = out["world_pred"]
            world_logvar = out.get("world_logvar")
            value_logvar = out.get("value_logvar")
            step_info = out.get("info_gain")
            if world_pred is None:
                break
            if world_pred.ndim == 3:
                obs_roll = world_pred[:, 0, :]
            else:
                obs_roll = world_pred
            if world_logvar is not None:
                if world_logvar.ndim == 3:
                    step_uncert = torch.exp(world_logvar[:, 0, :]).mean()
                else:
                    step_uncert = torch.exp(world_logvar).mean()
                uncertainty += float(step_uncert.item())
                uncertainty_steps += 1
            if value_logvar is not None:
                uncertainty += float(torch.exp(value_logvar).mean().item())
                uncertainty_steps += 1
            if step_info is not None:
                info_gain_score += float(step_info.mean().item())
                info_gain_steps += 1
            state_roll = out["state"]
            value = out["value"]
        if value is None:
            return -1e9
        score = float(value.squeeze().item())
        if uncertainty_steps > 0:
            score -= uncertainty_weight * (uncertainty / uncertainty_steps)
        if info_gain_steps > 0 and info_gain_weight != 0.0:
            score += info_gain_weight * (info_gain_score / info_gain_steps)
        return score

    def _plan_sample(
        self,
        *,
        action_logits: torch.Tensor,
        obs: torch.Tensor,
        task_ids: Optional[torch.Tensor],
        state: RecurrentState,
        num_candidates: int,
        horizon: int,
        uncertainty_weight: float,
        info_gain_weight: float,
        trace: bool = False,
    ) -> Dict[str, Any]:
        probs = torch.softmax(action_logits, dim=-1)
        candidates = torch.multinomial(probs, num_samples=num_candidates, replacement=True)
        batch = obs.shape[0]
        candidate_values = torch.full((batch, num_candidates), -1e9, device=obs.device, dtype=obs.dtype)
        trace_entries: list[dict[str, Any]] = []
        for b in range(batch):
            state_roll = self._slice_state(state, b)
            for c in range(num_candidates):
                action = candidates[b, c].view(1, 1)
                sequence = action.view(1).repeat(horizon)
                score = self._score_sequence(
                    obs[b : b + 1],
                    state_roll.clone(),
                    sequence,
                    task_ids=task_ids[b : b + 1] if task_ids is not None else None,
                    uncertainty_weight=uncertainty_weight,
                    info_gain_weight=info_gain_weight,
                )
                candidate_values[b, c] = score
            if trace:
                seqs = [
                    [int(candidates[b, c].item()) for _ in range(horizon)]
                    for c in range(num_candidates)
                ]
                trace_entries.append(
                    {
                        "candidates": seqs,
                        "scores": candidate_values[b].detach().cpu().tolist(),
                    }
                )
        best_idx = torch.argmax(candidate_values, dim=-1)
        best_actions = candidates.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)
        return {
            "action": best_actions,
            "action_logits": action_logits,
            "candidate_actions": candidates,
            "candidate_values": candidate_values,
            "trace": trace_entries if trace else None,
        }

    def _plan_cem(
        self,
        *,
        action_logits: torch.Tensor,
        obs: torch.Tensor,
        task_ids: Optional[torch.Tensor],
        state: RecurrentState,
        num_candidates: int,
        horizon: int,
        uncertainty_weight: float,
        info_gain_weight: float,
        cem_iters: int,
        elite_frac: float,
        trace: bool = False,
    ) -> Dict[str, Any]:
        batch = obs.shape[0]
        action_dim = action_logits.shape[-1]
        best_actions = torch.zeros(batch, dtype=torch.long, device=obs.device)
        candidate_values = torch.zeros(batch, num_candidates, device=obs.device, dtype=obs.dtype)
        candidates_out = torch.zeros(batch, num_candidates, horizon, dtype=torch.long, device=obs.device)
        trace_entries: list[dict[str, Any]] = []

        for b in range(batch):
            probs = torch.full((horizon, action_dim), 1.0 / action_dim, device=obs.device, dtype=obs.dtype)
            probs[0] = torch.softmax(action_logits[b], dim=-1)
            state_roll = self._slice_state(state, b)
            for _ in range(cem_iters):
                sequences = []
                for t in range(horizon):
                    seq_actions = torch.multinomial(probs[t], num_samples=num_candidates, replacement=True)
                    sequences.append(seq_actions)
                sequences_tensor = torch.stack(sequences, dim=1)
                scores = torch.tensor(
                    [
                        self._score_sequence(
                            obs[b : b + 1],
                            state_roll.clone(),
                            sequences_tensor[idx],
                            task_ids=task_ids[b : b + 1] if task_ids is not None else None,
                            uncertainty_weight=uncertainty_weight,
                            info_gain_weight=info_gain_weight,
                        )
                        for idx in range(num_candidates)
                    ],
                    device=obs.device,
                    dtype=obs.dtype,
                )
                elite_count = max(1, int(num_candidates * elite_frac))
                elite_idx = torch.topk(scores, k=elite_count).indices
                elite = sequences_tensor[elite_idx]
                for t in range(horizon):
                    counts = torch.bincount(elite[:, t], minlength=action_dim).float()
                    probs[t] = (counts / counts.sum()).clamp_min(1e-6)
                candidate_values[b] = scores
                candidates_out[b] = sequences_tensor

            best_idx = torch.argmax(candidate_values[b], dim=-1)
            best_actions[b] = candidates_out[b, best_idx, 0]
            if trace:
                trace_entries.append(
                    {
                        "candidates": candidates_out[b].detach().cpu().tolist(),
                        "scores": candidate_values[b].detach().cpu().tolist(),
                    }
                )

        return {
            "action": best_actions,
            "action_logits": action_logits,
            "candidate_actions": candidates_out,
            "candidate_values": candidate_values,
            "trace": trace_entries if trace else None,
        }

    def _plan_tree(
        self,
        *,
        action_logits: torch.Tensor,
        obs: torch.Tensor,
        task_ids: Optional[torch.Tensor],
        state: RecurrentState,
        horizon: int,
        uncertainty_weight: float,
        info_gain_weight: float,
        branch: int,
        trace: bool = False,
    ) -> Dict[str, Any]:
        batch = obs.shape[0]
        action_dim = action_logits.shape[-1]
        branch = min(branch, action_dim)
        best_actions = torch.zeros(batch, dtype=torch.long, device=obs.device)
        candidate_values = torch.full((batch, branch), -1e9, device=obs.device, dtype=obs.dtype)
        candidates = torch.zeros(batch, branch, dtype=torch.long, device=obs.device)
        trace_entries: list[dict[str, Any]] = []

        for b in range(batch):
            root_logits = action_logits[b]
            topk = torch.topk(root_logits, k=branch)
            candidates[b] = topk.indices
            for idx, action in enumerate(topk.indices):
                score = self._tree_rollout(
                    obs[b : b + 1],
                    self._slice_state(state, b),
                    action.view(1),
                    task_ids=task_ids[b : b + 1] if task_ids is not None else None,
                    depth=horizon,
                    branch=branch,
                    uncertainty_weight=uncertainty_weight,
                    info_gain_weight=info_gain_weight,
                )
                candidate_values[b, idx] = score
            if trace:
                trace_entries.append(
                    {
                        "candidates": candidates[b].detach().cpu().tolist(),
                        "scores": candidate_values[b].detach().cpu().tolist(),
                    }
                )
            best_idx = torch.argmax(candidate_values[b], dim=-1)
            best_actions[b] = candidates[b, best_idx]

        return {
            "action": best_actions,
            "action_logits": action_logits,
            "candidate_actions": candidates,
            "candidate_values": candidate_values,
            "trace": trace_entries if trace else None,
        }

    def _tree_rollout(
        self,
        obs: torch.Tensor,
        state: RecurrentState,
        action: torch.Tensor,
        *,
        task_ids: Optional[torch.Tensor],
        depth: int,
        branch: int,
        uncertainty_weight: float,
        info_gain_weight: float,
    ) -> float:
        action_tensor = action.view(1, 1)
        out = self.step(input_ids=action_tensor, obs=obs, state=state, task_ids=task_ids)
        world_pred = out["world_pred"]
        world_logvar = out.get("world_logvar")
        value_logvar = out.get("value_logvar")
        value = out["value"]
        info_gain = out.get("info_gain")

        uncertainty = 0.0
        uncertainty_steps = 0
        info_gain_score = 0.0
        info_gain_steps = 0
        if world_logvar is not None:
            if world_logvar.ndim == 3:
                uncertainty += float(torch.exp(world_logvar[:, 0, :]).mean().item())
            else:
                uncertainty += float(torch.exp(world_logvar).mean().item())
            uncertainty_steps += 1
        if value_logvar is not None:
            uncertainty += float(torch.exp(value_logvar).mean().item())
            uncertainty_steps += 1
        if info_gain is not None:
            info_gain_score += float(info_gain.mean().item())
            info_gain_steps += 1

        if depth <= 1 or world_pred is None:
            score = float(value.squeeze().item())
            if uncertainty_steps > 0:
                score -= uncertainty_weight * (uncertainty / uncertainty_steps)
            if info_gain_steps > 0 and info_gain_weight != 0.0:
                score += info_gain_weight * (info_gain_score / info_gain_steps)
            return score

        if world_pred.ndim == 3:
            obs_next = world_pred[:, 0, :]
        else:
            obs_next = world_pred

        next_logits = out["action_logits"].squeeze(0)
        branch = min(branch, next_logits.shape[-1])
        topk = torch.topk(next_logits, k=branch)
        best_score = -1e9
        for next_action in topk.indices:
            score = self._tree_rollout(
                obs_next,
                out["state"],
                next_action.view(1),
                task_ids=task_ids,
                depth=depth - 1,
                branch=branch,
                uncertainty_weight=uncertainty_weight,
                info_gain_weight=info_gain_weight,
            )
            if score > best_score:
                best_score = score
        return best_score

    @staticmethod
    def _mean_uncertainty(
        world_logvar: Optional[torch.Tensor],
        value_logvar: Optional[torch.Tensor],
    ) -> float:
        total = 0.0
        count = 0
        if world_logvar is not None:
            if world_logvar.ndim == 3:
                total += float(torch.exp(world_logvar[:, 0, :]).mean().item())
            else:
                total += float(torch.exp(world_logvar).mean().item())
            count += 1
        if value_logvar is not None:
            total += float(torch.exp(value_logvar).mean().item())
            count += 1
        if count == 0:
            return 0.0
        return total / count

    @staticmethod
    def _slice_state(state: RecurrentState, index: int) -> RecurrentState:
        mem = state.mem[index : index + 1].clone()
        keys = None
        values = None
        if state.kv.keys is not None:
            keys = [k[index : index + 1].clone() if k is not None else None for k in state.kv.keys]
        if state.kv.values is not None:
            values = [v[index : index + 1].clone() if v is not None else None for v in state.kv.values]
        kv = KVCache(keys=keys, values=values, max_len=state.kv.max_len)
        return RecurrentState(mem=mem, kv=kv, timestep=state.timestep)

    def _resolve_obs(
        self,
        obs: Optional[torch.Tensor],
        image: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if obs is not None:
            return obs
        if image is None:
            return None
        if self.vision is None:
            raise ValueError("image provided but use_vision is disabled")
        return self.vision(image)

    def _sanitize_obs(self, obs: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if obs is None:
            return None
        return sanitize_tensor(obs, "obs")

    def _format_world_logvar(self, h_pool: torch.Tensor) -> torch.Tensor:
        raw = self.world_logvar(h_pool) if self.world_logvar is not None else None
        if raw is None:
            raise ValueError("world_logvar head is not initialized")
        return raw.view(h_pool.shape[0], self.cfg.world_model_horizon, self.cfg.obs_dim)

    @staticmethod
    def _uncertainty_tensor(
        world_logvar: Optional[torch.Tensor],
        value_logvar: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if world_logvar is None and value_logvar is None:
            return torch.zeros((batch_size, 1), device=device)
        device = None
        if world_logvar is not None:
            device = world_logvar.device
        elif value_logvar is not None:
            device = value_logvar.device
        values = []
        if world_logvar is not None:
            if world_logvar.ndim == 3:
                world_var = torch.exp(world_logvar).mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
            else:
                world_var = torch.exp(world_logvar).mean(dim=-1, keepdim=True)
            values.append(world_var)
        if value_logvar is not None:
            value_var = torch.exp(value_logvar).mean(dim=-1, keepdim=True)
            values.append(value_var)
        if not values:
            return torch.zeros((batch_size, 1), device=device)
        return torch.stack(values, dim=0).mean(dim=0)

    @staticmethod
    def _confidence_from_uncertainty(uncertainty: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + uncertainty)

    def _budget_summary(
        self,
        mode_logits: Optional[torch.Tensor],
        horizon_logits: Optional[torch.Tensor],
        candidate_logits: Optional[torch.Tensor],
    ) -> Dict[str, int | str]:
        return self._budget_from_logits(mode_logits, horizon_logits, candidate_logits)

    def _apply_obs_uncertainty(
        self,
        logvar: Optional[torch.Tensor],
        obs: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if logvar is None or obs is None or self.cfg.uncertainty_obs_scale <= 0.0:
            return logvar
        obs_var = obs.var(dim=-1, keepdim=True) * self.cfg.uncertainty_obs_scale
        if logvar.ndim == 2:
            return logvar + obs_var
        if logvar.ndim == 3:
            scaled = obs_var[:, None, :].expand(-1, logvar.shape[1], logvar.shape[2])
            return logvar + scaled
        return logvar
