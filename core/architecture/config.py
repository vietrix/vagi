"""
vAGI v2.0 Architecture Configuration

Hyperparameters for:
- Multi-Head Latent Attention (MLA)
- Mixture of Experts (MoE)
- MCTS Reasoning Engine
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class MLAConfig:
    """
    Multi-Head Latent Attention Configuration.

    MLA compresses Key-Value states into a low-rank latent space,
    reducing KV cache from O(n_heads * d_head * seq_len) to O(d_latent * seq_len).

    Memory savings: ~(n_heads * d_head) / d_latent times smaller KV cache.
    """
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA groups
    head_dim: int = 128

    # Low-rank compression dimensions (the key innovation of MLA)
    # d_latent << n_kv_heads * head_dim for significant compression
    kv_latent_dim: int = 512  # Compressed KV dimension
    q_latent_dim: int = 1536  # Query latent (can be larger for expressiveness)

    # RoPE settings
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    max_position_embeddings: int = 131072  # 128K context

    # Attention settings
    attention_dropout: float = 0.0
    attention_bias: bool = False

    # Decoupled RoPE for position-aware queries
    qk_rope_head_dim: int = 64  # Separate RoPE dimension
    qk_nope_head_dim: int = 64  # Non-positional dimension


@dataclass
class MoEConfig:
    """
    Mixture of Experts Configuration.

    Sparse MoE activates only top-k experts per token, providing:
    - Linear scaling of parameters without linear compute cost
    - Specialization of experts for different domains

    Shared experts (DeepSeek-V3 innovation) maintain core knowledge
    that should be active for all tokens.
    """
    hidden_size: int = 4096
    intermediate_size: int = 14336  # FFN intermediate dim

    # Expert configuration
    num_experts: int = 64  # Total routed experts
    num_shared_experts: int = 2  # Always-active experts
    num_experts_per_tok: int = 6  # Top-K routing

    # Router settings
    router_aux_loss_coef: float = 0.001  # Load balancing loss weight
    router_z_loss_coef: float = 0.001  # Router z-loss for stability
    router_jitter_noise: float = 0.0  # Noise for exploration during training

    # Expert capacity (for load balancing)
    expert_capacity_factor: float = 1.25

    # Normalization
    norm_topk_prob: bool = True  # Normalize top-k probabilities

    # FFN activation
    hidden_act: str = "silu"


@dataclass
class MCTSConfig:
    """
    Monte Carlo Tree Search Configuration for System 2 Reasoning.

    MCTS explores reasoning paths before committing to an answer,
    using UCT (Upper Confidence Bound for Trees) for selection.

    UCT formula: Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
    where c is the exploration constant.
    """
    # Tree search parameters
    num_simulations: int = 50  # MCTS iterations per decision
    max_depth: int = 10  # Maximum reasoning depth

    # UCT exploration constant (higher = more exploration)
    c_puct: float = 1.414  # sqrt(2) is theoretically optimal

    # Expansion settings
    num_expansions: int = 5  # Number of thoughts to generate per expansion
    temperature: float = 0.8  # Sampling temperature for expansion

    # Value estimation
    value_weight: float = 0.5  # Weight of value vs. rollout
    discount_factor: float = 0.99  # Future value discount

    # Pruning
    min_visit_count: int = 1  # Minimum visits before pruning consideration
    prune_threshold: float = 0.01  # Prune nodes with very low value

    # Parallelization
    virtual_loss: float = 1.0  # Virtual loss for parallel MCTS
    num_parallel: int = 4  # Parallel simulations


@dataclass
class CodeInterpreterConfig:
    """
    Code Interpreter Configuration for Code-as-Reasoning.

    Instead of hallucinating computation, the model writes and
    executes code to verify facts and perform calculations.
    """
    # Execution limits
    max_execution_time: float = 30.0  # Seconds
    max_memory_mb: int = 512  # Memory limit
    max_output_length: int = 10000  # Truncate long outputs

    # Sandbox settings
    sandbox_type: Literal["subprocess", "docker", "restricted"] = "restricted"
    allowed_modules: list = field(default_factory=lambda: [
        "math", "statistics", "collections", "itertools", "functools",
        "re", "json", "datetime", "random", "string", "decimal",
        "fractions", "heapq", "bisect", "copy", "operator"
    ])

    # Code parsing
    code_block_pattern: str = r"```(?:python)?\n(.*?)```"

    # Feedback format
    include_traceback: bool = True
    max_traceback_lines: int = 10


@dataclass
class VAGIv2Config:
    """
    Complete vAGI v2.0 Configuration.

    Combines MLA, MoE, MCTS for a complete reasoning engine.
    """
    # Model dimensions
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    vocab_size: int = 102400

    # Sub-configs
    mla: MLAConfig = field(default_factory=MLAConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    code_interpreter: CodeInterpreterConfig = field(default_factory=CodeInterpreterConfig)

    # General settings
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    use_cache: bool = True

    # Training settings
    initializer_range: float = 0.02

    # Reasoning mode
    enable_mcts: bool = True
    enable_code_interpreter: bool = True

    def __post_init__(self):
        """Sync dimensions across sub-configs."""
        self.mla.hidden_size = self.hidden_size
        self.moe.hidden_size = self.hidden_size

    @classmethod
    def from_model_size(cls, size: Literal["7B", "22B", "70B", "236B"]) -> "VAGIv2Config":
        """Create config for standard model sizes."""
        configs = {
            "7B": dict(
                hidden_size=4096,
                num_hidden_layers=32,
                mla=MLAConfig(num_attention_heads=32, num_key_value_heads=8),
                moe=MoEConfig(num_experts=64, num_experts_per_tok=6),
            ),
            "22B": dict(
                hidden_size=6144,
                num_hidden_layers=48,
                mla=MLAConfig(
                    hidden_size=6144,
                    num_attention_heads=48,
                    num_key_value_heads=8,
                    kv_latent_dim=768
                ),
                moe=MoEConfig(
                    hidden_size=6144,
                    num_experts=128,
                    num_experts_per_tok=8
                ),
            ),
            "70B": dict(
                hidden_size=8192,
                num_hidden_layers=80,
                mla=MLAConfig(
                    hidden_size=8192,
                    num_attention_heads=64,
                    num_key_value_heads=8,
                    kv_latent_dim=1024
                ),
                moe=MoEConfig(
                    hidden_size=8192,
                    num_experts=256,
                    num_shared_experts=4,
                    num_experts_per_tok=8
                ),
            ),
            "236B": dict(
                hidden_size=12288,
                num_hidden_layers=96,
                mla=MLAConfig(
                    hidden_size=12288,
                    num_attention_heads=96,
                    num_key_value_heads=12,
                    kv_latent_dim=1536,
                    q_latent_dim=3072
                ),
                moe=MoEConfig(
                    hidden_size=12288,
                    intermediate_size=24576,
                    num_experts=512,
                    num_shared_experts=8,
                    num_experts_per_tok=12
                ),
            ),
        }
        return cls(**configs[size])
