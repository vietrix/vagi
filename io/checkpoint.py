"""Checkpoint save/load helpers using safetensors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file, save_file


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    out_dir: str | Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save model weights (safetensors) and optimizer state."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.safetensors"
    save_file(model.state_dict(), str(model_path))

    meta = {"step": int(step), "extra": extra or {}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), out_dir / "optimizer.pt")
    return model_path


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ckpt_path: str | Path = "model.safetensors",
) -> Dict[str, Any]:
    """Load a checkpoint from a directory or safetensors file."""
    ckpt_path = Path(ckpt_path)
    base_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
    meta_path = base_dir / "meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    model_path = ckpt_path
    if ckpt_path.is_dir():
        model_path = ckpt_path / "model.safetensors"

    state = load_file(str(model_path))
    model.load_state_dict(state, strict=True)

    if optimizer is not None:
        optimizer_path = base_dir / "optimizer.pt"
        if optimizer_path.exists():
            optimizer_state = torch.load(optimizer_path, map_location="cpu")
            optimizer.load_state_dict(optimizer_state)

    return meta
