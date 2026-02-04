"""
vAGI v2.0 Architecture Module.

This module provides the core components for efficient inference:
- MLA (Multi-Head Latent Attention): Low-rank KV compression
- MoE (Mixture of Experts): Sparse conditional computation
- RoPE (Rotary Position Embeddings): Position-aware attention
- Adaptive Operations: Hardware-aware kernel selection
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

from .ops import (
    DeviceManager,
    DeviceCapabilities,
    get_device_manager,
    adaptive_attention,
    adaptive_matmul,
    adaptive_layer_norm,
    AdaptiveAttention,
    get_attention_backend,
    to_device,
)

from .moe_router import (
    BalancedTopKRouter,
    BalancedMoELayer,
    SharedExpertMoE,
    MoEMetrics,
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
    # Balanced MoE
    "BalancedTopKRouter",
    "BalancedMoELayer",
    "SharedExpertMoE",
    "MoEMetrics",
    # Adaptive Operations
    "DeviceManager",
    "DeviceCapabilities",
    "get_device_manager",
    "adaptive_attention",
    "adaptive_matmul",
    "adaptive_layer_norm",
    "AdaptiveAttention",
    "get_attention_backend",
    "to_device",
    # Model
    "RMSNorm",
    "VAGIDecoderLayer",
    "VAGIModel",
    "VAGIForCausalLM",
]
