"""Main vAGI model."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from .backbone import CausalTransformerBackbone
from .config import VAGIConfig
from .heads import LanguageHead, LogVarHead, PolicyHead, ValueHead, WorldHead
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
        self.value_logvar = LogVarHead(cfg.hidden_size, 1) if cfg.use_uncertainty else None
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

        x, h_last, h_act, mem_next = self.backbone(
            input_ids=input_ids,
            obs=obs,
            task_ids=task_ids,
            state=state,
        )

        text_logits = self.lang(x)
        h_policy = h_act if h_act is not None else h_last
        action_logits = self.pi(h_policy)
        value = self.v(h_policy)
        world_pred = self.world(h_last) if self.world is not None else None
        value_logvar = self.value_logvar(h_policy) if self.value_logvar is not None else None
        world_logvar = self._format_world_logvar(h_last) if self.world_logvar is not None else None
        value_logvar = self._apply_obs_uncertainty(value_logvar, obs)
        world_logvar = self._apply_obs_uncertainty(world_logvar, obs)

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
            "state": state_out,
        }
        if return_hidden:
            outputs["hidden"] = h_last

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
                losses["value"] = value_loss(value, targets["values"], logvar=value_logvar)
            if world_pred is not None:
                if "obs_future" in targets:
                    losses["world"] = world_loss(world_pred, targets["obs_future"], logvar=world_logvar)
                elif "obs_next" in targets:
                    losses["world"] = world_loss(world_pred, targets["obs_next"], logvar=world_logvar)

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
        obs: torch.Tensor,
        state: RecurrentState,
        task_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Single-step inference with recurrent state update."""
        if state is None:
            raise ValueError("state is required for step()")
        if obs is None:
            raise ValueError("obs is required for step()")
        x, h_last, h_act, mem_next, kv_next = self.backbone.forward_step(
            input_ids=input_ids,
            obs=obs,
            task_ids=task_ids,
            state=state,
        )
        h_policy = h_act if h_act is not None else h_last
        outputs: Dict[str, Any] = {
            "text_logits": self.lang(x),
            "action_logits": self.pi(h_policy),
            "value": self.v(h_policy),
            "value_logvar": None,
            "world_pred": self.world(h_last) if self.world is not None else None,
            "world_logvar": None,
            "state": None,
        }
        if self.value_logvar is not None:
            outputs["value_logvar"] = self._apply_obs_uncertainty(self.value_logvar(h_policy), obs)
        if self.world_logvar is not None:
            outputs["world_logvar"] = self._apply_obs_uncertainty(self._format_world_logvar(h_last), obs)
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
        task_ids: Optional[torch.Tensor] = None,
        num_candidates: int = 4,
        horizon: int = 3,
        uncertainty_weight: float = 1.0,
        strategy: str = "cem",
        cem_iters: int = 3,
        elite_frac: float = 0.2,
        tree_branching: int = 4,
        uncertainty_fallback: Optional[float] = None,
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
        if obs is None:
            raise ValueError("obs is required for plan_step")

        base = self.forward(
            input_ids=input_ids,
            obs=obs,
            task_ids=task_ids,
            state=state,
            return_loss=False,
        )
        action_logits = base["action_logits"]
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
                }

        strategy_key = strategy.lower().strip()
        if strategy_key == "sample":
            return self._plan_sample(
                action_logits=action_logits,
                obs=obs,
                task_ids=task_ids,
                state=state,
                num_candidates=num_candidates,
                horizon=horizon,
                uncertainty_weight=uncertainty_weight,
            )
        if strategy_key == "tree":
            return self._plan_tree(
                action_logits=action_logits,
                obs=obs,
                task_ids=task_ids,
                state=state,
                horizon=horizon,
                uncertainty_weight=uncertainty_weight,
                branch=tree_branching,
            )
        return self._plan_cem(
            action_logits=action_logits,
            obs=obs,
            task_ids=task_ids,
            state=state,
            num_candidates=num_candidates,
            horizon=horizon,
            uncertainty_weight=uncertainty_weight,
            cem_iters=cem_iters,
            elite_frac=elite_frac,
        )

    def _score_sequence(
        self,
        obs: torch.Tensor,
        state: RecurrentState,
        sequence: torch.Tensor,
        *,
        task_ids: Optional[torch.Tensor],
        uncertainty_weight: float,
    ) -> float:
        obs_roll = obs
        state_roll = state
        value = None
        uncertainty = 0.0
        uncertainty_steps = 0
        for action in sequence:
            action_tensor = action.view(1, 1)
            out = self.step(input_ids=action_tensor, obs=obs_roll, state=state_roll, task_ids=task_ids)
            world_pred = out["world_pred"]
            world_logvar = out.get("world_logvar")
            value_logvar = out.get("value_logvar")
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
            state_roll = out["state"]
            value = out["value"]
        if value is None:
            return -1e9
        score = float(value.squeeze().item())
        if uncertainty_steps > 0:
            score -= uncertainty_weight * (uncertainty / uncertainty_steps)
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
    ) -> Dict[str, Any]:
        probs = torch.softmax(action_logits, dim=-1)
        candidates = torch.multinomial(probs, num_samples=num_candidates, replacement=True)
        batch = obs.shape[0]
        candidate_values = torch.full((batch, num_candidates), -1e9, device=obs.device, dtype=obs.dtype)
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
                )
                candidate_values[b, c] = score
        best_idx = torch.argmax(candidate_values, dim=-1)
        best_actions = candidates.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)
        return {
            "action": best_actions,
            "action_logits": action_logits,
            "candidate_actions": candidates,
            "candidate_values": candidate_values,
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
        cem_iters: int,
        elite_frac: float,
    ) -> Dict[str, Any]:
        batch = obs.shape[0]
        action_dim = action_logits.shape[-1]
        best_actions = torch.zeros(batch, dtype=torch.long, device=obs.device)
        candidate_values = torch.zeros(batch, num_candidates, device=obs.device, dtype=obs.dtype)
        candidates_out = torch.zeros(batch, num_candidates, horizon, dtype=torch.long, device=obs.device)

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

        return {
            "action": best_actions,
            "action_logits": action_logits,
            "candidate_actions": candidates_out,
            "candidate_values": candidate_values,
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
        branch: int,
    ) -> Dict[str, Any]:
        batch = obs.shape[0]
        action_dim = action_logits.shape[-1]
        branch = min(branch, action_dim)
        best_actions = torch.zeros(batch, dtype=torch.long, device=obs.device)
        candidate_values = torch.full((batch, branch), -1e9, device=obs.device, dtype=obs.dtype)
        candidates = torch.zeros(batch, branch, dtype=torch.long, device=obs.device)

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
                )
                candidate_values[b, idx] = score
            best_idx = torch.argmax(candidate_values[b], dim=-1)
            best_actions[b] = candidates[b, best_idx]

        return {
            "action": best_actions,
            "action_logits": action_logits,
            "candidate_actions": candidates,
            "candidate_values": candidate_values,
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
    ) -> float:
        action_tensor = action.view(1, 1)
        out = self.step(input_ids=action_tensor, obs=obs, state=state, task_ids=task_ids)
        world_pred = out["world_pred"]
        world_logvar = out.get("world_logvar")
        value_logvar = out.get("value_logvar")
        value = out["value"]

        uncertainty = 0.0
        uncertainty_steps = 0
        if world_logvar is not None:
            if world_logvar.ndim == 3:
                uncertainty += float(torch.exp(world_logvar[:, 0, :]).mean().item())
            else:
                uncertainty += float(torch.exp(world_logvar).mean().item())
            uncertainty_steps += 1
        if value_logvar is not None:
            uncertainty += float(torch.exp(value_logvar).mean().item())
            uncertainty_steps += 1

        if depth <= 1 or world_pred is None:
            score = float(value.squeeze().item())
            if uncertainty_steps > 0:
                score -= uncertainty_weight * (uncertainty / uncertainty_steps)
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

    def _format_world_logvar(self, h_last: torch.Tensor) -> torch.Tensor:
        raw = self.world_logvar(h_last) if self.world_logvar is not None else None
        if raw is None:
            raise ValueError("world_logvar head is not initialized")
        return raw.view(h_last.shape[0], self.cfg.world_model_horizon, self.cfg.obs_dim)

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
