"""Checkpoint utilities using safetensors for vAGI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file, save_file

from vagi_core import VAGIConfig


def _to_device_str(device: torch.device | str) -> str:
    return str(device) if isinstance(device, torch.device) else device


def save_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[VAGIConfig] = None,
    step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
    model_filename: str = "model.safetensors",
) -> Path:
    """Save model weights with safetensors plus optional optimizer/meta."""
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    model_path = path / model_filename
    save_file(model.state_dict(), str(model_path))

    meta = {
        "step": int(step),
        "config": config.__dict__ if config is not None else None,
        "extra": extra or {},
        "model_file": model_filename,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if optimizer is not None:
        torch.save(optimizer.state_dict(), path / "optimizer.pt")

    return path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint into an existing model (and optional optimizer)."""
    path = Path(checkpoint_path)
    base_dir = path if path.is_dir() else path.parent
    meta_path = base_dir / "meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if path.is_dir():
        model_filename = meta.get("model_file", "model.safetensors")
        model_path = path / model_filename
    else:
        model_path = path

    state = load_file(str(model_path), device=_to_device_str(device))
    model.load_state_dict(state, strict=True)

    if optimizer is not None:
        optimizer_path = base_dir / "optimizer.pt"
        if optimizer_path.exists():
            optimizer_state = torch.load(optimizer_path, map_location=device)
            optimizer.load_state_dict(optimizer_state)

    return meta


def load_config_from_checkpoint(checkpoint_path: str | Path) -> Optional[VAGIConfig]:
    path = Path(checkpoint_path)
    base_dir = path if path.is_dir() else path.parent
    meta_path = base_dir / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg_dict = meta.get("config")
    if not cfg_dict:
        return None
    return VAGIConfig(**cfg_dict)
