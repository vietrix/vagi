from __future__ import annotations

import torch

from vagi_core import VAGIConfig, VAGICore


def test_init_state_prefill_kv_shapes() -> None:
    cfg = VAGIConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=8,
        obs_tokens=1,
        action_dim=4,
        memory_slots=2,
        dropout=0.0,
        use_gqa=True,
    )
    model = VAGICore(cfg)
    state = model.init_state(batch_size=2, prefill_kv=True, kv_max_seq_len=4)
    assert state.kv.keys is not None
    assert state.kv.values is not None
    assert state.kv.max_len == 4
    head_dim = cfg.hidden_size // cfg.n_heads
    for k in state.kv.keys:
        assert isinstance(k, torch.Tensor)
        assert k.shape == (2, cfg.n_kv_heads, 0, head_dim)
    for v in state.kv.values:
        assert isinstance(v, torch.Tensor)
        assert v.shape == (2, cfg.n_kv_heads, 0, head_dim)
