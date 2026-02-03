"""Backbone components for vAGI."""

from __future__ import annotations

from typing import Optional, Tuple
from contextlib import nullcontext

import torch
from torch.utils.checkpoint import checkpoint
from torch import nn
from torch.nn import functional as F

from .config import VAGIConfig
from .memory import KVCache, RecurrentState
from .utils import check_floating, check_shape, StageTimer


class ObsTokenizer(nn.Module):
    """Project observations into a fixed number of tokens with learnable positional encoding."""

    def __init__(self, obs_dim: int, obs_tokens: int, hidden_size: int) -> None:
        super().__init__()
        self.obs_tokens = obs_tokens
        self.hidden_size = hidden_size
        self.proj = nn.Linear(obs_dim, obs_tokens * hidden_size)
        # Learnable positional encoding for obs tokens (Issue 2.3)
        self.obs_pos_embed = nn.Parameter(torch.zeros(1, obs_tokens, hidden_size))
        nn.init.normal_(self.obs_pos_embed, std=0.02)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        check_shape(obs, (None, None), "obs")
        check_floating(obs, "obs")
        bsz = obs.shape[0]
        projected = self.proj(obs)
        tokens = projected.view(bsz, self.obs_tokens, self.hidden_size)
        # Add learnable positional encoding
        tokens = tokens + self.obs_pos_embed
        return tokens


def gumbel_softmax_binary(logits: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
    """Gumbel-softmax for differentiable binary decisions (Issue 2.4).

    Args:
        logits: Input logits of shape (...,)
        temperature: Gumbel temperature. Lower = harder decisions
        hard: If True, use straight-through estimator for hard decisions

    Returns:
        Soft or hard binary decisions in [0, 1]
    """
    if temperature <= 0:
        temperature = 1e-6
    # Sample Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-10) + 1e-10) + 1e-10)
    # Soft sample
    y_soft = torch.sigmoid((logits + gumbel_noise) / temperature)
    if hard:
        # Straight-through estimator: hard in forward, soft in backward
        y_hard = (y_soft > 0.5).float()
        return y_hard - y_soft.detach() + y_soft
    return y_soft


class FastMemory(nn.Module):
    """Fast memory update module with gated writes, Gumbel-softmax erase gates, and stabilization."""

    def __init__(
        self,
        hidden_size: int,
        memory_slots: int,
        *,
        decay: float,
        protect: bool,
        consolidate_every: int,
        erase_temperature: float = 1.0,
        erase_hard: bool = False,
    ) -> None:
        super().__init__()
        self.memory_slots = memory_slots
        self.decay = decay
        self.consolidate_every = consolidate_every
        # Gumbel-softmax temperature for erase gates (Issue 2.4)
        self.erase_temperature = erase_temperature
        self.erase_hard = erase_hard
        # Round-robin consolidation slot tracking (Issue 2.5)
        self.register_buffer("consolidate_slot_idx", torch.tensor(0, dtype=torch.long))
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.write = nn.Linear(hidden_size, hidden_size)
        self.erase = nn.Linear(hidden_size, memory_slots)
        self.protect = nn.Linear(hidden_size, 1) if protect else None
        self.consolidate_proj = nn.Linear(hidden_size, hidden_size) if consolidate_every > 0 else None
        self.consolidate_gate = nn.Linear(hidden_size, 1) if consolidate_every > 0 else None
        # Importance weighting for slot selection (Issue 2.5)
        self.slot_importance = nn.Linear(hidden_size, 1) if consolidate_every > 0 else None

    def set_erase_temperature(self, temperature: float) -> None:
        """Set Gumbel-softmax temperature for erase gates (for annealing during training)."""
        self.erase_temperature = max(1e-6, temperature)

    def set_erase_hard(self, hard: bool) -> None:
        """Set whether to use hard binary decisions for erase gates."""
        self.erase_hard = hard

    def forward(
        self,
        mem: torch.Tensor,
        h_last: torch.Tensor,
        timestep: Optional[int] = None,
        erase_temperature: Optional[float] = None,
    ) -> torch.Tensor:
        check_shape(mem, (None, self.memory_slots, None), "mem")
        if self.memory_slots == 0:
            return mem
        if self.decay < 1.0:
            mem = mem * self.decay

        q = self.query(h_last)  # (B, D)
        k = self.key(mem)  # (B, M, D)
        v = self.value(mem)  # (B, M, D)

        scale = k.shape[-1] ** 0.5
        attn_scores = (k * q.unsqueeze(1)).sum(dim=-1) / scale
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, M)

        write = self.write(h_last)  # (B, D)
        # Use Gumbel-softmax for differentiable binary erase decisions (Issue 2.4)
        erase_logits = self.erase(h_last)  # (B, M)
        temp = erase_temperature if erase_temperature is not None else self.erase_temperature
        erase = gumbel_softmax_binary(erase_logits, temperature=temp, hard=self.erase_hard)

        write_update = attn_weights.unsqueeze(-1) * write.unsqueeze(1)
        erase_mask = erase.unsqueeze(-1) * attn_weights.unsqueeze(-1)
        if self.protect is not None:
            protect_gate = torch.sigmoid(self.protect(mem))  # (B, M, 1)
            write_update = write_update * (1.0 - protect_gate)
            erase_mask = erase_mask * (1.0 - protect_gate)
        mem = mem * (1.0 - erase_mask) + write_update
        _ = (attn_weights.unsqueeze(-1) * v).sum(dim=1)
        if self.consolidate_every > 0 and timestep is not None:
            if (timestep + 1) % self.consolidate_every == 0:
                mem = self._consolidate(mem)
        return mem

    def _consolidate(self, mem: torch.Tensor) -> torch.Tensor:
        """Consolidate memory with round-robin or importance-weighted slot assignment (Issue 2.5)."""
        if self.consolidate_proj is None or self.consolidate_gate is None:
            raise ValueError("consolidation modules not initialized")

        summary = mem.mean(dim=1, keepdim=True)
        gate = torch.sigmoid(self.consolidate_gate(summary))
        compressed = self.consolidate_proj(summary)

        # Determine target slot using importance weighting or round-robin
        if self.slot_importance is not None:
            # Importance-weighted slot selection: pick slot with lowest importance
            importance_scores = self.slot_importance(mem).squeeze(-1)  # (B, M)
            # Add small noise for tie-breaking
            importance_scores = importance_scores + torch.randn_like(importance_scores) * 0.01
            # Select slot with minimum importance (least important to overwrite)
            target_slot = torch.argmin(importance_scores, dim=-1)  # (B,)
        else:
            # Round-robin slot assignment
            target_slot = self.consolidate_slot_idx.expand(mem.shape[0])
            # Update round-robin counter (only during training)
            if self.training:
                self.consolidate_slot_idx = (self.consolidate_slot_idx + 1) % self.memory_slots

        # Apply consolidation to selected slots
        mem = mem * (1.0 - gate)
        batch_size = mem.shape[0]
        for b in range(batch_size):
            slot_idx = target_slot[b].item() if target_slot.dim() > 0 else int(target_slot.item())
            mem[b, slot_idx:slot_idx+1, :] = mem[b, slot_idx:slot_idx+1, :] + gate[b] * compressed[b]

        return mem


SPECIAL_TOKENS = ("<ACT>",)


class FeedForward(nn.Module):
    """Simple MLP block with GELU activation."""

    def __init__(self, hidden_size: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        mid = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, mid)
        self.fc2 = nn.Linear(mid, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class CausalSelfAttention(nn.Module):
    """Causal self-attention with optional KV cache."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float,
        use_gqa: bool,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if use_gqa else n_heads
        self.head_dim = hidden_size // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_size, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, hidden_size, bias=False)

    def _reshape_q(self, q: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = q.shape
        return q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _reshape_kv(self, kv: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = kv.shape
        return kv.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

    def _expand_kv(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n_kv_heads == self.n_heads:
            return k, v
        repeat = self.n_heads // self.n_kv_heads
        return k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)

    @staticmethod
    def _step_mask(past_len: int, new_len: int, device: torch.device) -> torch.Tensor:
        if past_len < 0:
            raise ValueError("past_len must be >= 0")
        if new_len <= 0:
            raise ValueError("new_len must be > 0")
        if past_len == 0:
            return torch.triu(torch.ones(new_len, new_len, device=device, dtype=torch.bool), diagonal=1)
        q_pos = torch.arange(new_len, device=device).unsqueeze(1)
        k_pos = torch.arange(past_len + new_len, device=device).unsqueeze(0)
        return k_pos > (past_len + q_pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._reshape_q(self.q_proj(x))
        k = self._reshape_kv(self.k_proj(x))
        v = self._reshape_kv(self.v_proj(x))
        k, v = self._expand_kv(k, v)
        dropout_p = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        bsz, _, seq_len, _ = attn.shape
        out = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.out_proj(out)

    def forward_step(
        self,
        x: torch.Tensor,
        kv_cache: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        q = self._reshape_q(self.q_proj(x))
        k_new = self._reshape_kv(self.k_proj(x))
        v_new = self._reshape_kv(self.v_proj(x))

        k_prev, v_prev = kv_cache
        if k_prev is None:
            k_cat = k_new
            v_cat = v_new
            past_len = 0
        else:
            k_cat = torch.cat([k_prev, k_new], dim=2)
            v_cat = torch.cat([v_prev, v_new], dim=2)
            past_len = k_prev.shape[2]

        k_attn, v_attn = self._expand_kv(k_cat, v_cat)
        mask = self._step_mask(past_len, x.shape[1], x.device)
        dropout_p = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=mask, dropout_p=dropout_p)
        bsz, _, seq_len, _ = attn.shape
        out = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.out_proj(out), (k_cat, v_cat)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm attention and MLP."""

    def __init__(self, cfg: VAGIConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = CausalSelfAttention(
            hidden_size=cfg.hidden_size,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            dropout=cfg.dropout,
            use_gqa=cfg.use_gqa,
        )
        self.ln2 = nn.LayerNorm(cfg.hidden_size)
        self.mlp = FeedForward(cfg.hidden_size, cfg.mlp_ratio, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_step(
        self,
        x: torch.Tensor,
        kv_cache: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, kv_next = self.attn.forward_step(self.ln1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, kv_next


class CausalTransformerBackbone(nn.Module):
    """Causal transformer decoder backbone with optional observation tokens."""

    def __init__(self, cfg: VAGIConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        extra_tokens = len(SPECIAL_TOKENS) if cfg.use_special_tokens else 0
        self.pos_embed = nn.Embedding(cfg.max_seq_len + cfg.obs_tokens + extra_tokens, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.obs_tokens = cfg.obs_tokens
        self.use_special_tokens = cfg.use_special_tokens
        self.special_embed = nn.Embedding(len(SPECIAL_TOKENS), cfg.hidden_size) if cfg.use_special_tokens else None
        self.obs_tokenizer = ObsTokenizer(cfg.obs_dim, cfg.obs_tokens, cfg.hidden_size) if cfg.obs_tokens > 0 else None
        self.task_embed = nn.Embedding(cfg.task_vocab_size, cfg.hidden_size) if cfg.use_task_embedding else None
        self.memory = (
            FastMemory(
                cfg.hidden_size,
                cfg.memory_slots,
                decay=cfg.memory_decay,
                protect=cfg.memory_protect,
                consolidate_every=cfg.memory_consolidate_every,
                erase_temperature=cfg.memory_erase_temperature,
                erase_hard=cfg.memory_erase_hard,
            )
            if cfg.memory_slots > 0
            else None
        )
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])

    def _combine_inputs(
        self,
        token_embed: torch.Tensor,
        obs_tokens: Optional[torch.Tensor],
        include_special: bool,
    ) -> Tuple[torch.Tensor, Optional[int]]:
        bsz = token_embed.shape[0]
        chunks = []
        act_index = None
        if obs_tokens is not None:
            chunks.append(obs_tokens)
        chunks.append(token_embed)
        if include_special and self.special_embed is not None:
            act_token = self.special_embed.weight.unsqueeze(0).expand(bsz, -1, -1)
            chunks.append(act_token)
            act_index = sum(chunk.shape[1] for chunk in chunks) - 1
        x = torch.cat(chunks, dim=1)
        return x, act_index

    def _add_positions(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        seq_len = x.shape[1]
        pos_ids = torch.arange(seq_len, device=x.device) + start_pos
        pos_ids = pos_ids % self.pos_embed.num_embeddings
        x = x + self.pos_embed(pos_ids).unsqueeze(0)
        return self.dropout(x)

    @staticmethod
    def _kv_len(kv: Optional[KVCache]) -> int:
        if kv is None or kv.keys is None or not kv.keys:
            return 0
        if kv.keys[0] is None:
            return 0
        return int(kv.keys[0].shape[2])

    def _should_checkpoint_layer(self, layer_idx: int) -> bool:
        """Determine if a specific layer should use gradient checkpointing (Issue 2.12).

        Args:
            layer_idx: Index of the transformer layer

        Returns:
            True if gradient checkpointing should be used for this layer
        """
        if not self.cfg.use_grad_checkpoint:
            return False
        # If grad_checkpoint_layers is specified, only checkpoint those layers
        if self.cfg.grad_checkpoint_layers is not None:
            return layer_idx in self.cfg.grad_checkpoint_layers
        # Otherwise checkpoint all layers
        return True

    def forward(
        self,
        input_ids: torch.Tensor,
        obs: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
        state: Optional[RecurrentState] = None,
        timer: Optional[StageTimer] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must be torch.long")
        check_shape(input_ids, (None, None), "input_ids")

        with timer.track("embed") if timer else nullcontext():
            token_embed = self.token_embed(input_ids)
            obs_tokens = None
            include_special = self.use_special_tokens
            if obs is not None and self.obs_tokenizer is not None:
                obs_tokens = self.obs_tokenizer(obs)

        x, act_index = self._combine_inputs(token_embed, obs_tokens, include_special)
        if task_ids is not None and self.task_embed is not None:
            if task_ids.dtype != torch.long:
                raise TypeError("task_ids must be torch.long")
            if task_ids.ndim != 1 or task_ids.shape[0] != input_ids.shape[0]:
                raise ValueError("task_ids must have shape (B,)")
            task_emb = self.task_embed(task_ids).unsqueeze(1)
            x = x + task_emb

        seq_len = x.shape[1]
        if seq_len > self.pos_embed.num_embeddings:
            raise ValueError("Sequence length exceeds configured max_seq_len + obs_tokens")

        with timer.track("attn") if timer else nullcontext():
            x = self._add_positions(x, start_pos=0)
            for idx, block in enumerate(self.blocks):
                # Per-layer gradient checkpointing (Issue 2.12)
                use_checkpoint = self._should_checkpoint_layer(idx)
                if use_checkpoint and self.training:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)

        h_last = x[:, -1, :]
        h_act = x[:, act_index, :] if act_index is not None else None

        mem_next = None
        if state is not None and self.memory is not None:
            h_pool = h_act if h_act is not None else h_last
            with timer.track("mem") if timer else nullcontext():
                mem_next = self.memory(state.mem, h_pool, timestep=state.timestep)

        return x, h_last, h_act, mem_next

    def forward_step(
        self,
        input_ids: torch.Tensor,
        obs: torch.Tensor,
        task_ids: Optional[torch.Tensor],
        state: RecurrentState,
        timer: Optional[StageTimer] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], KVCache]:
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must be torch.long")
        check_shape(input_ids, (None, None), "input_ids")
        if obs is None:
            raise ValueError("obs is required for forward_step")

        with timer.track("embed") if timer else nullcontext():
            token_embed = self.token_embed(input_ids)
            obs_tokens = self.obs_tokenizer(obs) if self.obs_tokenizer is not None else None
            include_special = self.use_special_tokens
            x, act_index = self._combine_inputs(token_embed, obs_tokens, include_special)
            if task_ids is not None and self.task_embed is not None:
                if task_ids.dtype != torch.long:
                    raise TypeError("task_ids must be torch.long")
                if task_ids.ndim != 1 or task_ids.shape[0] != input_ids.shape[0]:
                    raise ValueError("task_ids must have shape (B,)")
                task_emb = self.task_embed(task_ids).unsqueeze(1)
                x = x + task_emb

        start_pos = self._kv_len(state.kv)
        with timer.track("attn") if timer else nullcontext():
            x = self._add_positions(x, start_pos=start_pos)

        kv_next = KVCache.empty(len(self.blocks))
        max_len = state.kv.max_len if state.kv is not None else None
        new_len = x.shape[1]
        with timer.track("attn") if timer else nullcontext():
            for idx, block in enumerate(self.blocks):
                k_prev = None
                v_prev = None
                if state.kv.keys is not None:
                    k_prev = state.kv.keys[idx]
                if state.kv.values is not None:
                    v_prev = state.kv.values[idx]
                if max_len is not None and k_prev is not None and v_prev is not None:
                    allowed_past = max_len - new_len
                    if allowed_past < 0:
                        allowed_past = 0
                    if k_prev.shape[2] > allowed_past:
                        if allowed_past == 0:
                            k_prev = k_prev[:, :, :0, :]
                            v_prev = v_prev[:, :, :0, :]
                        else:
                            k_prev = k_prev[:, :, -allowed_past:, :]
                            v_prev = v_prev[:, :, -allowed_past:, :]
                x, (k_new, v_new) = block.forward_step(x, (k_prev, v_prev))
                kv_next.keys[idx] = k_new
                kv_next.values[idx] = v_new
        if max_len is not None:
            kv_next.max_len = max_len

        h_last = x[:, -1, :]
        h_act = x[:, act_index, :] if act_index is not None else None

        mem_next = None
        if self.memory is not None:
            h_pool = h_act if h_act is not None else h_last
            with timer.track("mem") if timer else nullcontext():
                mem_next = self.memory(state.mem, h_pool, timestep=state.timestep)

        return x, h_last, h_act, mem_next, kv_next
