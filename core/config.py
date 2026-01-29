"""Configuration for vAGI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VAGIConfig:
    """Model configuration values."""

    vocab_size: int = 256
    hidden_size: int = 128
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 4
    mlp_ratio: float = 4.0
    max_seq_len: int = 256
    obs_dim: int = 16
    obs_tokens: int = 2
    action_dim: int = 8
    memory_slots: int = 4
    dropout: float = 0.1
    use_rotary: bool = False
    use_gqa: bool = False
    use_flash_attn: bool = False
    use_world_pred: bool = False
    world_model_horizon: int = 1
    use_confidence: bool = False
    use_uncertainty: bool = False
    uncertainty_obs_scale: float = 0.0
    use_special_tokens: bool = True

    def __post_init__(self) -> None:
        if self.use_confidence and not self.use_uncertainty:
            self.use_uncertainty = True
        self.validate()

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be > 0")
        if self.hidden_size % self.n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")
        if self.n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be > 0")
        if self.use_gqa and (self.n_heads % self.n_kv_heads != 0):
            raise ValueError("n_heads must be divisible by n_kv_heads when use_gqa is True")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be > 0")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        if self.obs_dim <= 0:
            raise ValueError("obs_dim must be > 0")
        if self.obs_tokens < 0:
            raise ValueError("obs_tokens must be >= 0")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if self.memory_slots < 0:
            raise ValueError("memory_slots must be >= 0")
        if self.world_model_horizon <= 0:
            raise ValueError("world_model_horizon must be > 0")
        if self.uncertainty_obs_scale < 0.0:
            raise ValueError("uncertainty_obs_scale must be >= 0")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
