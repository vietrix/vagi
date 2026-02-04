"""
vAGI v2.0 Architecture Module.

This module provides the core components for efficient inference:
- MLA (Multi-Head Latent Attention): Low-rank KV compression
- MoE (Mixture of Experts): Sparse conditional computation
- RoPE (Rotary Position Embeddings): Position-aware attention
"""

from .config import (
    MLAConfig,
    MoEConfig,
    MCTSConfig,
    CodeInterpreterConfig,
    VAGIv2Config,
)

from .modeling_vagi import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
    MLAAttention,
    Expert,
    MoELayer,
    RMSNorm,
    VAGIDecoderLayer,
    VAGIModel,
    VAGIForCausalLM,
)

from .router import (
    TopKRouter,
    ExpertChoiceRouter,
    SwitchRouter,
    SoftMoERouter,
    load_balancing_loss,
    router_z_loss,
)


__all__ = [
    # Configs
    "MLAConfig",
    "MoEConfig",
    "MCTSConfig",
    "CodeInterpreterConfig",
    "VAGIv2Config",
    # Attention & Position
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
    "MLAAttention",
    # MoE
    "Expert",
    "MoELayer",
    "TopKRouter",
    "ExpertChoiceRouter",
    "SwitchRouter",
    "SoftMoERouter",
    "load_balancing_loss",
    "router_z_loss",
    # Model
    "RMSNorm",
    "VAGIDecoderLayer",
    "VAGIModel",
    "VAGIForCausalLM",
]
