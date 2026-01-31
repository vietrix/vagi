"""Model configuration presets for vAGI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .config import VAGIConfig

VAGI_LITE_DEFAULTS: Dict[str, object] = {
    "vocab_size": 256,
    "hidden_size": 64,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 4,
    "mlp_ratio": 2.0,
    "max_seq_len": 128,
    "obs_dim": 16,
    "obs_tokens": 2,
    "action_dim": 8,
    "memory_slots": 4,
    "dropout": 0.1,
    "use_rotary": False,
    "use_gqa": False,
    "use_flash_attn": False,
    "use_world_pred": False,
    "use_special_tokens": True,
}

VAGI_LITE_PATH = Path(__file__).resolve().parent / "vagi_lite.json"


def load_vagi_lite_config(path: Path | None = None) -> VAGIConfig:
    """Load the vAGI-lite preset from disk or fall back to defaults."""
    config_path = path or VAGI_LITE_PATH
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return VAGIConfig(**payload)
    return VAGIConfig(**VAGI_LITE_DEFAULTS)


def save_vagi_lite_config(cfg: VAGIConfig, path: Path | None = None) -> Path:
    """Persist a vAGI-lite preset for later reuse."""
    config_path = path or VAGI_LITE_PATH
    payload = cfg.__dict__.copy()
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config_path

