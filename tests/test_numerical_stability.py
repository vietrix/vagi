"""Numerical stability tests."""

from __future__ import annotations

import pytest
import torch

from core.base import VAGIConfig, VAGICore


def _build_model(max_seq_len: int = 8) -> VAGICore:
    cfg = VAGIConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=max_seq_len,
        obs_dim=8,
        obs_tokens=2,
        action_dim=6,
        memory_slots=2,
        dropout=0.0,
        use_world_pred=True,
    )
    return VAGICore(cfg).eval()


def test_nan_inf_obs_sanitized() -> None:
    model = _build_model()
    input_ids = torch.ones((1, 2), dtype=torch.long)
    obs = torch.tensor([[float("nan"), float("inf"), -float("inf"), 1.0, -1.0, 0.0, 3.0, -3.0]])

    out = model.forward(input_ids=input_ids, obs=obs, state=None, return_loss=False)
    assert torch.isfinite(out["text_logits"]).all()
    assert torch.isfinite(out["action_logits"]).all()
    assert torch.isfinite(out["value"]).all()
    assert out["world_pred"] is None or torch.isfinite(out["world_pred"]).all()


def test_seq_len_bounds() -> None:
    model = _build_model(max_seq_len=4)
    obs = torch.zeros((1, model.cfg.obs_dim), dtype=torch.float32)

    with pytest.raises(ValueError, match="sequence length"):
        model.forward(input_ids=torch.zeros((1, 0), dtype=torch.long), obs=obs, state=None, return_loss=False)

    with pytest.raises(ValueError, match="max_seq_len"):
        model.forward(input_ids=torch.zeros((1, 5), dtype=torch.long), obs=obs, state=None, return_loss=False)
