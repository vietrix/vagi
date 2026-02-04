"""
vAGI v2.0 Core Architecture: MoE + MLA Implementation

This module implements the architectural backbone for efficient inference:
- Multi-Head Latent Attention (MLA): Low-rank KV compression for memory efficiency
- Sparse Mixture of Experts (MoE): Conditional computation for scale
- Rotary Position Embeddings (RoPE): Position-aware attention

Mathematical Foundations:

MLA (Multi-Head Latent Attention):
    Standard MHA requires storing K, V of shape [batch, seq, n_heads, head_dim].
    MLA compresses this to a shared latent c_kv of shape [batch, seq, d_latent].

    Key insight: K, V can be reconstructed from low-rank projections:
        K = c_kv @ W_uk  (up-projection from latent)
        V = c_kv @ W_uv

    This reduces KV cache from O(n_heads * d_head) to O(d_latent) per token.
    For DeepSeek-V2: 13.5x memory reduction on long contexts.

MoE (Mixture of Experts):
    Instead of one FFN, use many "expert" FFNs with sparse activation.
    Router g(x) selects top-k experts per token:

        y = Σ_i g_i(x) * Expert_i(x)  for top-k experts

    Shared experts (always active) maintain core knowledge stability.
"""

import math
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import MLAConfig, MoEConfig, VAGIv2Config
from .router import TopKRouter, load_balancing_loss


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    RoPE encodes position by rotating query and key vectors:
        q' = R(θ, m) @ q
        k' = R(θ, n) @ k

    where R(θ, pos) is a rotation matrix and θ_i = 10000^(-2i/d).

    Benefits:
    - Relative position awareness: q'·k' depends on (m-n)
    - Extrapolates to longer sequences than training
    - No learnable parameters
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor

        # Precompute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache for cos/sin values
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cos/sin cache if needed."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len

            # Position indices
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            t = t / self.scaling_factor

            # Compute frequencies: [seq_len, dim/2]
            freqs = torch.outer(t, self.inv_freq.to(device))

            # Duplicate for pairs: [seq_len, dim]
            emb = torch.cat((freqs, freqs), dim=-1)

            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(
        self,
        x: Tensor,
        position_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute rotary embeddings for given positions.

        Args:
            x: Input tensor [batch, seq, heads, head_dim]
            position_ids: Position indices [batch, seq]

        Returns:
            (cos, sin) tensors for rotation
        """
        seq_len = x.shape[1]
        self._update_cache(seq_len, x.device, x.dtype)

        if position_ids is not None:
            # Gather cos/sin for specific positions
            cos = self._cos_cached[position_ids]  # [batch, seq, dim]
            sin = self._sin_cached[position_ids]
        else:
            cos = self._cos_cached[:seq_len]  # [seq, dim]
            sin = self._sin_cached[:seq_len]

        return cos, sin


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    unsqueeze_dim: int = 2,
) -> Tuple[Tensor, Tensor]:
    """
    Apply rotary position embedding to query and key tensors.

    Args:
        q: Query tensor [batch, seq, heads, head_dim]
        k: Key tensor [batch, seq, kv_heads, head_dim]
        cos: Cosine component [batch, seq, head_dim] or [seq, head_dim]
        sin: Sine component [batch, seq, head_dim] or [seq, head_dim]
        unsqueeze_dim: Dimension to unsqueeze for broadcasting

    Returns:
        Rotated (q, k) tensors
    """
    # Expand for head dimension
    if cos.dim() == 2:
        cos = cos.unsqueeze(unsqueeze_dim)  # [seq, 1, dim] or [batch, seq, 1, dim]
        sin = sin.unsqueeze(unsqueeze_dim)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

    # Apply rotation: x' = x * cos + rotate_half(x) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class MLAAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA).

    MLA achieves KV cache compression through low-rank factorization:

    Standard Attention:
        Q = x @ W_q  -> [batch, seq, n_heads * head_dim]
        K = x @ W_k  -> [batch, seq, n_kv_heads * head_dim]
        V = x @ W_v  -> [batch, seq, n_kv_heads * head_dim]

    MLA Attention:
        c_kv = x @ W_dkv           -> [batch, seq, d_latent]  (compress)
        K = c_kv @ W_uk            -> [batch, seq, n_kv_heads * head_dim]  (decompress)
        V = c_kv @ W_uv            -> [batch, seq, n_kv_heads * head_dim]  (decompress)

    During inference, we only cache c_kv instead of K and V.
    Memory reduction: (n_kv_heads * head_dim) / d_latent times smaller.

    Additional innovation: Decoupled RoPE
        - Split Q, K into RoPE and non-RoPE components
        - Allows position information without corrupting the latent space
    """

    def __init__(self, config: MLAConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.kv_latent_dim = config.kv_latent_dim
        self.q_latent_dim = config.q_latent_dim

        # Decoupled RoPE dimensions
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim

        # Number of heads per KV group (for GQA)
        self.num_heads_per_kv = self.num_heads // self.num_kv_heads

        # ============ Query Projections ============
        # Q uses latent compression for memory efficiency during training
        # q_latent = x @ W_dq  (down-project)
        # Q = q_latent @ W_uq  (up-project)
        self.q_down_proj = nn.Linear(
            self.hidden_size,
            self.q_latent_dim,
            bias=config.attention_bias,
        )
        self.q_up_proj = nn.Linear(
            self.q_latent_dim,
            self.num_heads * self.qk_nope_head_dim,
            bias=config.attention_bias,
        )

        # Separate projection for RoPE queries
        self.q_rope_proj = nn.Linear(
            self.q_latent_dim,
            self.num_heads * self.qk_rope_head_dim,
            bias=config.attention_bias,
        )

        # ============ KV Projections (The Core Innovation) ============
        # Compress to latent: c_kv = x @ W_dkv
        self.kv_down_proj = nn.Linear(
            self.hidden_size,
            self.kv_latent_dim,
            bias=config.attention_bias,
        )

        # Decompress K (non-RoPE): K_nope = c_kv @ W_uk
        self.k_up_proj = nn.Linear(
            self.kv_latent_dim,
            self.num_kv_heads * self.qk_nope_head_dim,
            bias=config.attention_bias,
        )

        # Separate RoPE keys: K_rope = c_kv @ W_uk_rope
        self.k_rope_proj = nn.Linear(
            self.kv_latent_dim,
            self.num_kv_heads * self.qk_rope_head_dim,
            bias=config.attention_bias,
        )

        # Decompress V: V = c_kv @ W_uv
        self.v_up_proj = nn.Linear(
            self.kv_latent_dim,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )

        # ============ Output Projection ============
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        # ============ RoPE ============
        self.rotary_emb = RotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Attention scaling
        # Use combined head dim for scaling
        self.scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5

        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass with MLA attention.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, kv_seq_len]
            position_ids: [batch, seq_len]
            past_key_value: Cached (c_kv_latent, k_rope) for efficient generation
            output_attentions: Whether to return attention weights
            use_cache: Whether to return updated cache

        Returns:
            (output, attention_weights, past_key_value)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # ============ Query Computation ============
        q_latent = self.q_down_proj(hidden_states)

        # Non-RoPE queries: [batch, seq, num_heads, qk_nope_head_dim]
        q_nope = self.q_up_proj(q_latent)
        q_nope = q_nope.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim)

        # RoPE queries: [batch, seq, num_heads, qk_rope_head_dim]
        q_rope = self.q_rope_proj(q_latent)
        q_rope = q_rope.view(batch_size, seq_len, self.num_heads, self.qk_rope_head_dim)

        # ============ KV Computation ============
        # Compress to latent (this is what we cache!)
        c_kv = self.kv_down_proj(hidden_states)  # [batch, seq, kv_latent_dim]

        # Decompress K (non-RoPE): [batch, seq, num_kv_heads, qk_nope_head_dim]
        k_nope = self.k_up_proj(c_kv)
        k_nope = k_nope.view(batch_size, seq_len, self.num_kv_heads, self.qk_nope_head_dim)

        # RoPE keys: [batch, seq, num_kv_heads, qk_rope_head_dim]
        k_rope = self.k_rope_proj(c_kv)
        k_rope = k_rope.view(batch_size, seq_len, self.num_kv_heads, self.qk_rope_head_dim)

        # Values: [batch, seq, num_kv_heads, head_dim]
        v = self.v_up_proj(c_kv)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # ============ Apply RoPE ============
        cos, sin = self.rotary_emb(q_rope, position_ids)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)

        # ============ Handle KV Cache ============
        if past_key_value is not None:
            # past_key_value = (cached_c_kv, cached_k_rope)
            cached_c_kv, cached_k_rope = past_key_value

            # Concatenate with cached values
            c_kv = torch.cat([cached_c_kv, c_kv], dim=1)
            k_rope = torch.cat([cached_k_rope, k_rope], dim=1)

            # Recompute k_nope and v from full c_kv
            k_nope = self.k_up_proj(c_kv)
            k_nope = k_nope.view(batch_size, -1, self.num_kv_heads, self.qk_nope_head_dim)

            v = self.v_up_proj(c_kv)
            v = v.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Update cache
        if use_cache:
            # Only cache the latent c_kv and rotary keys
            past_key_value = (c_kv, k_rope)
        else:
            past_key_value = None

        # ============ Combine Q, K components ============
        # Concatenate RoPE and non-RoPE components
        q = torch.cat([q_nope, q_rope], dim=-1)  # [batch, seq, heads, nope+rope]
        k = torch.cat([k_nope, k_rope], dim=-1)  # [batch, kv_seq, kv_heads, nope+rope]

        # ============ Grouped Query Attention ============
        kv_seq_len = k.shape[1]

        # Expand KV heads for GQA: [batch, kv_seq, num_heads, head_dim]
        if self.num_heads_per_kv > 1:
            k = k.repeat_interleave(self.num_heads_per_kv, dim=2)
            v = v.repeat_interleave(self.num_heads_per_kv, dim=2)

        # Transpose for attention: [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ============ Scaled Dot-Product Attention ============
        # attn_weights: [batch, heads, seq, kv_seq]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)

        # Compute output: [batch, heads, seq, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # Reshape: [batch, seq, num_heads * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Expert(nn.Module):
    """
    Single MoE Expert (SwiGLU FFN).

    SwiGLU: out = (gate(x) * Swish(gate(x))) @ W_down

    Better than standard FFN for language modeling.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """SwiGLU forward pass."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    """
    Mixture of Experts Layer with Shared Experts.

    Architecture:
        y = Shared_Experts(x) + Σ_i g_i(x) * Expert_i(x)

    Shared experts are always active, maintaining core knowledge.
    Routed experts are activated top-k per token.

    Load Balancing:
        We use auxiliary losses to ensure balanced expert usage:
        - load_balancing_loss: Penalize uneven routing
        - router_z_loss: Stabilize router logits
    """

    def __init__(self, config: MoEConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.num_shared_experts = config.num_shared_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # ============ Shared Experts (Always Active) ============
        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                Expert(self.hidden_size, self.intermediate_size)
                for _ in range(self.num_shared_experts)
            ])
        else:
            self.shared_experts = None

        # ============ Routed Experts ============
        self.experts = nn.ModuleList([
            Expert(self.hidden_size, self.intermediate_size)
            for _ in range(self.num_experts)
        ])

        # ============ Router ============
        self.router = TopKRouter(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            aux_loss_coef=config.router_aux_loss_coef,
            z_loss_coef=config.router_z_loss_coef,
            jitter_noise=config.router_jitter_noise,
            norm_topk_prob=config.norm_topk_prob,
        )

        # Store auxiliary loss for training
        self.aux_loss = None

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through MoE layer.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            (output, aux_loss)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Flatten for routing: [batch * seq, hidden_size]
        hidden_flat = hidden_states.view(-1, hidden_size)

        # ============ Shared Expert Output ============
        if self.shared_experts is not None:
            shared_output = sum(
                expert(hidden_flat) for expert in self.shared_experts
            ) / self.num_shared_experts
        else:
            shared_output = 0

        # ============ Route to Experts ============
        # router_output: (expert_weights, expert_indices, aux_loss)
        # expert_weights: [batch * seq, num_experts_per_tok]
        # expert_indices: [batch * seq, num_experts_per_tok]
        expert_weights, expert_indices, aux_loss = self.router(hidden_flat)
        self.aux_loss = aux_loss

        # ============ Compute Expert Outputs ============
        # Efficient batched expert computation
        routed_output = torch.zeros_like(hidden_flat)

        # Group tokens by expert for batched computation
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)

            if not expert_mask.any():
                continue

            # Get tokens and their weights for this expert
            token_indices = expert_mask.nonzero(as_tuple=True)[0]
            tokens = hidden_flat[token_indices]

            # Get the weight for this expert (where it was selected)
            weight_mask = expert_indices[token_indices] == expert_idx
            weights = (expert_weights[token_indices] * weight_mask.float()).sum(dim=-1, keepdim=True)

            # Compute expert output
            expert_out = self.experts[expert_idx](tokens)

            # Add weighted output
            routed_output[token_indices] += weights * expert_out

        # ============ Combine Outputs ============
        output = shared_output + routed_output

        # Reshape back
        output = output.view(batch_size, seq_len, hidden_size)

        return output, aux_loss


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    Simpler and often faster than LayerNorm.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class VAGIDecoderLayer(nn.Module):
    """
    Single decoder layer with MLA attention and MoE FFN.

    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + MoE(RMSNorm(x))
    """

    def __init__(
        self,
        config: VAGIv2Config,
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Attention with MLA
        self.self_attn = MLAAttention(config.mla, layer_idx)

        # MoE FFN
        self.mlp = MoELayer(config.moe, layer_idx)

        # Normalization
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor, Tensor]], Optional[Tensor]]:
        """
        Forward pass through decoder layer.

        Returns:
            (hidden_states, attention_weights, past_key_value, moe_aux_loss)
        """
        residual = hidden_states

        # Pre-norm + Attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights, past_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Pre-norm + MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, moe_aux_loss = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights, past_key_value, moe_aux_loss


class VAGIModel(nn.Module):
    """
    vAGI v2.0 Transformer Model.

    Full architecture with:
    - Token embeddings
    - Stacked decoder layers (MLA + MoE)
    - Final RMSNorm
    """

    def __init__(self, config: VAGIv2Config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Decoder layers
        self.layers = nn.ModuleList([
            VAGIDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Gradient checkpointing
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> dict:
        """
        Forward pass through the model.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            position_ids: [batch, seq_len]
            past_key_values: List of cached KV states
            use_cache: Whether to return caches
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states

        Returns:
            Dictionary with:
            - last_hidden_state: [batch, seq_len, hidden_size]
            - past_key_values: Updated caches
            - hidden_states: All layer outputs (if requested)
            - attentions: All attention weights (if requested)
            - moe_aux_loss: Combined auxiliary loss from MoE layers
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Prepare position IDs
        if position_ids is None:
            if past_key_values is not None and past_key_values[0] is not None:
                # Continue from cached position
                past_len = past_key_values[0][0].shape[1]
                position_ids = torch.arange(
                    past_len, past_len + seq_len,
                    device=input_ids.device
                ).unsqueeze(0).expand(batch_size, -1)
            else:
                position_ids = torch.arange(
                    seq_len, device=input_ids.device
                ).unsqueeze(0).expand(batch_size, -1)

        # Prepare attention mask
        if attention_mask is not None:
            # Create causal mask
            causal_mask = self._prepare_causal_mask(
                attention_mask, seq_len, past_key_values
            )
        else:
            causal_mask = None

        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_cache = () if use_cache else None
        total_moe_loss = 0.0

        # Forward through layers
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values else None

            if self.gradient_checkpointing and self.training:
                hidden_states, attn_weights, present_key_value, moe_loss = (
                    self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_value,
                        output_attentions,
                        use_cache,
                    )
                )
            else:
                hidden_states, attn_weights, present_key_value, moe_loss = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            if use_cache:
                next_cache += (present_key_value,)

            if output_attentions:
                all_attentions += (attn_weights,)

            if moe_loss is not None:
                total_moe_loss += moe_loss

        # Final norm
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache if use_cache else None,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
            "moe_aux_loss": total_moe_loss / self.config.num_hidden_layers,
        }

    def _prepare_causal_mask(
        self,
        attention_mask: Tensor,
        query_length: int,
        past_key_values: Optional[List],
    ) -> Tensor:
        """Prepare 4D causal attention mask."""
        batch_size = attention_mask.shape[0]

        # Get total sequence length including cache
        if past_key_values is not None and past_key_values[0] is not None:
            past_length = past_key_values[0][0].shape[1]
        else:
            past_length = 0

        key_length = past_length + query_length

        # Create causal mask
        causal_mask = torch.triu(
            torch.full((query_length, key_length), float("-inf"), device=attention_mask.device),
            diagonal=past_length + 1,
        )

        # Expand to 4D: [batch, 1, query_len, key_len]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.expand(batch_size, 1, -1, -1)

        # Apply padding mask
        if attention_mask.dim() == 2:
            # Expand padding mask: [batch, 1, 1, key_len]
            padding_mask = attention_mask[:, None, None, :]
            padding_mask = (1.0 - padding_mask) * float("-inf")

            # Expand for past if needed
            if past_length > 0:
                past_mask = torch.zeros(
                    batch_size, 1, 1, past_length,
                    device=attention_mask.device
                )
                padding_mask = torch.cat([past_mask, padding_mask], dim=-1)

            causal_mask = causal_mask + padding_mask

        return causal_mask


class VAGIForCausalLM(nn.Module):
    """
    vAGI v2.0 for Causal Language Modeling.

    Adds language modeling head on top of VAGIModel.
    """

    def __init__(self, config: VAGIv2Config):
        super().__init__()
        self.config = config

        # Base model
        self.model = VAGIModel(config)

        # LM head (optionally tied with embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        labels: Optional[Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> dict:
        """
        Forward pass with optional loss computation.

        Args:
            labels: [batch, seq_len] for computing language modeling loss

        Returns:
            Dictionary with loss (if labels provided) and logits
        """
        # Get model outputs
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs["last_hidden_state"]

        # Compute logits
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

            # Add MoE auxiliary loss
            if outputs["moe_aux_loss"] is not None:
                loss = loss + outputs["moe_aux_loss"]

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs["past_key_values"],
            "hidden_states": outputs["hidden_states"],
            "attentions": outputs["attentions"],
            "moe_aux_loss": outputs["moe_aux_loss"],
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        use_cache: bool = True,
        **kwargs,
    ) -> Tensor:
        """
        Generate text autoregressively.

        Simple greedy/sampling generation. For reasoning tasks,
        use MCTS engine instead.
        """
        batch_size = input_ids.shape[0]
        past_key_values = None

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(
                input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

            logits = outputs["logits"][:, -1, :]  # [batch, vocab]
            past_key_values = outputs["past_key_values"]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample or greedy
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Check for EOS (assuming EOS token id is 2)
            if (next_token == 2).all():
                break

        return input_ids
