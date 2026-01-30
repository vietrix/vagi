"""Serving configuration for vAGI API."""

from __future__ import annotations

from dataclasses import dataclass
import os

from vagi_core import VAGIConfig


@dataclass
class ServeConfig:
    model_id: str = "vagi"
    device: str = "cpu"
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
    use_world_pred: bool = False
    world_model_horizon: int = 1
    use_uncertainty: bool = False

    def build_model_config(self) -> VAGIConfig:
        return VAGIConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            mlp_ratio=self.mlp_ratio,
            max_seq_len=self.max_seq_len,
            obs_dim=self.obs_dim,
            obs_tokens=self.obs_tokens,
            action_dim=self.action_dim,
            memory_slots=self.memory_slots,
            dropout=self.dropout,
            use_world_pred=self.use_world_pred,
            world_model_horizon=self.world_model_horizon,
            use_uncertainty=self.use_uncertainty,
        )


def load_config() -> ServeConfig:
    return ServeConfig(
        model_id=os.getenv("VAGI_MODEL_ID", "vagi"),
        device=os.getenv("VAGI_DEVICE", "cpu"),
        vocab_size=int(os.getenv("VAGI_VOCAB_SIZE", "256")),
        hidden_size=int(os.getenv("VAGI_HIDDEN_SIZE", "128")),
        n_layers=int(os.getenv("VAGI_LAYERS", "2")),
        n_heads=int(os.getenv("VAGI_HEADS", "4")),
        n_kv_heads=int(os.getenv("VAGI_KV_HEADS", "4")),
        mlp_ratio=float(os.getenv("VAGI_MLP_RATIO", "4.0")),
        max_seq_len=int(os.getenv("VAGI_MAX_SEQ_LEN", "256")),
        obs_dim=int(os.getenv("VAGI_OBS_DIM", "16")),
        obs_tokens=int(os.getenv("VAGI_OBS_TOKENS", "2")),
        action_dim=int(os.getenv("VAGI_ACTION_DIM", "8")),
        memory_slots=int(os.getenv("VAGI_MEMORY_SLOTS", "4")),
        dropout=float(os.getenv("VAGI_DROPOUT", "0.1")),
        use_world_pred=os.getenv("VAGI_USE_WORLD", "0") == "1",
        world_model_horizon=int(os.getenv("VAGI_WORLD_HORIZON", "1")),
        use_uncertainty=os.getenv("VAGI_USE_UNCERTAINTY", "0") == "1",
    )
