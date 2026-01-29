"""Main vAGI model."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from .backbone import CausalTransformerBackbone
from .config import VAGIConfig
from .heads import ConfidenceHead, LanguageHead, PolicyHead, ValueHead, WorldHead
from .losses import language_loss, policy_loss, total_loss, value_loss, world_loss
from .memory import KVCache, RecurrentState
from .utils import check_floating, check_shape


class VAGICore(nn.Module):
    """vAGI causal transformer with recurrent state."""

    def __init__(self, cfg: VAGIConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = CausalTransformerBackbone(cfg)
        self.lang = LanguageHead(cfg.hidden_size, cfg.vocab_size)
        self.pi = PolicyHead(cfg.hidden_size, cfg.action_dim)
        self.v = ValueHead(cfg.hidden_size)
        self.value_conf = ConfidenceHead(cfg.hidden_size, 1) if cfg.use_confidence else None
        self.world = (
            WorldHead(cfg.hidden_size, cfg.obs_dim, horizon=cfg.world_model_horizon)
            if cfg.use_world_pred
            else None
        )
        self.world_conf = (
            ConfidenceHead(cfg.hidden_size, cfg.world_model_horizon)
            if cfg.use_confidence and cfg.use_world_pred
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
        state: Optional[RecurrentState] = None,
        labels: Optional[torch.Tensor] = None,
        targets: Optional[Dict[str, Any]] = None,
        return_loss: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass for full sequences."""
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must be torch.long")
        check_shape(input_ids, (None, None), "input_ids")

        if obs is not None:
            check_shape(obs, (input_ids.shape[0], self.cfg.obs_dim), "obs")
            check_floating(obs, "obs")

        if state is not None:
            if not isinstance(state, RecurrentState):
                raise TypeError("state must be a RecurrentState")
            check_shape(state.mem, (input_ids.shape[0], self.cfg.memory_slots, self.cfg.hidden_size), "state.mem")

        x, h_last, h_act, mem_next = self.backbone(input_ids=input_ids, obs=obs, state=state)

        text_logits = self.lang(x)
        h_policy = h_act if h_act is not None else h_last
        action_logits = self.pi(h_policy)
        value = self.v(h_policy)
        value_conf = self.value_conf(h_policy) if self.value_conf is not None else None
        world_pred = self.world(h_last) if self.world is not None else None
        world_conf = self.world_conf(h_last) if self.world_conf is not None else None

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
            "value_conf": value_conf,
            "world_pred": world_pred,
            "world_conf": world_conf,
            "state": state_out,
        }

        if return_loss:
            losses: Dict[str, torch.Tensor] = {}
            if labels is not None:
                if labels.dtype != torch.long:
                    raise TypeError("labels must be torch.long")
                include_special = self.cfg.use_special_tokens and obs is not None
                k_prefix = self.cfg.obs_tokens + (1 if include_special else 0) if obs is not None else 0
                k_suffix = 2 if include_special else 0
                losses["language"] = language_loss(text_logits, labels, k_prefix=k_prefix, k_suffix=k_suffix)
            targets = targets or {}
            if "actions" in targets:
                losses["policy"] = policy_loss(action_logits, targets["actions"])
            if "values" in targets:
                losses["value"] = value_loss(value, targets["values"])
            if world_pred is not None:
                if "obs_future" in targets:
                    losses["world"] = world_loss(world_pred, targets["obs_future"])
                elif "obs_next" in targets:
                    losses["world"] = world_loss(world_pred, targets["obs_next"])

            if losses:
                outputs["loss"] = total_loss(losses, weights=targets.get("loss_weights"))
                outputs["losses_breakdown"] = losses
            else:
                outputs["loss"] = None
                outputs["losses_breakdown"] = {}

        return outputs

    @torch.no_grad()
    def step(self, input_ids: torch.Tensor, obs: torch.Tensor, state: RecurrentState) -> Dict[str, Any]:
        """Single-step inference with recurrent state update."""
        if state is None:
            raise ValueError("state is required for step()")
        if obs is None:
            raise ValueError("obs is required for step()")
        x, h_last, h_act, mem_next, kv_next = self.backbone.forward_step(
            input_ids=input_ids,
            obs=obs,
            state=state,
        )
        h_policy = h_act if h_act is not None else h_last
        outputs: Dict[str, Any] = {
            "text_logits": self.lang(x),
            "action_logits": self.pi(h_policy),
            "value": self.v(h_policy),
            "value_conf": self.value_conf(h_policy) if self.value_conf is not None else None,
            "world_pred": self.world(h_last) if self.world is not None else None,
            "world_conf": self.world_conf(h_last) if self.world_conf is not None else None,
            "state": None,
        }
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
    def plan_step(
        self,
        input_ids: torch.Tensor,
        obs: torch.Tensor,
        state: RecurrentState,
        num_candidates: int = 4,
        horizon: int = 3,
        confidence_weight: float = 1.0,
    ) -> Dict[str, Any]:
        """Plan a single action by rolling out the world model and scoring with value."""
        if num_candidates <= 0:
            raise ValueError("num_candidates must be > 0")
        if horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.world is None:
            raise ValueError("world model is required for plan_step")
        if state is None:
            raise ValueError("state is required for plan_step")
        if obs is None:
            raise ValueError("obs is required for plan_step")

        base = self.forward(input_ids=input_ids, obs=obs, state=state, return_loss=False)
        action_logits = base["action_logits"]
        probs = torch.softmax(action_logits, dim=-1)
        candidates = torch.multinomial(probs, num_samples=num_candidates, replacement=True)

        batch = obs.shape[0]
        candidate_values = torch.full((batch, num_candidates), -1e9, device=obs.device, dtype=obs.dtype)
        for b in range(batch):
            for c in range(num_candidates):
                action = candidates[b, c].view(1, 1)
                obs_roll = obs[b : b + 1]
                state_roll = self._slice_state(state, b)
                value = None
                conf_score = None
                for _ in range(horizon):
                    out = self.step(input_ids=action, obs=obs_roll, state=state_roll)
                    world_pred = out["world_pred"]
                    world_conf = out.get("world_conf")
                    if world_pred is None:
                        break
                    if world_pred.ndim == 3:
                        obs_roll = world_pred[:, 0, :]
                    else:
                        obs_roll = world_pred
                    if world_conf is not None:
                        if world_conf.ndim == 2:
                            conf_score = world_conf.mean()
                        else:
                            conf_score = world_conf.mean()
                    state_roll = out["state"]
                    value = out["value"]
                if value is not None:
                    score = value.squeeze()
                    if conf_score is not None:
                        score = score - confidence_weight * (1.0 - conf_score)
                    candidate_values[b, c] = score

        best_idx = torch.argmax(candidate_values, dim=-1)
        best_actions = candidates.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)
        return {
            "action": best_actions,
            "action_logits": action_logits,
            "candidate_actions": candidates,
            "candidate_values": candidate_values,
        }

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
