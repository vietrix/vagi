import torch

from vagi_core import VAGIConfig, VAGICore


def _make_cfg() -> VAGIConfig:
    return VAGIConfig(
        vocab_size=128,
        hidden_size=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=32,
        obs_dim=16,
        obs_tokens=2,
        action_dim=8,
        memory_slots=4,
        dropout=0.0,
        use_world_pred=True,
    )


def test_shapes_with_obs() -> None:
    cfg = _make_cfg()
    model = VAGICore(cfg)
    bsz, tlen = 2, 5
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, tlen), dtype=torch.long)
    obs = torch.randn(bsz, cfg.obs_dim)
    state = model.init_state(bsz)

    out = model.forward(input_ids=input_ids, obs=obs, state=state)
    extra = 3 if cfg.use_special_tokens and obs is not None else 0
    seq_len = tlen + cfg.obs_tokens + extra

    assert out["text_logits"].shape == (bsz, seq_len, cfg.vocab_size)
    assert out["action_logits"].shape == (bsz, cfg.action_dim)
    assert out["value"].shape == (bsz, 1)
    assert out["world_pred"] is not None
    assert out["world_pred"].shape == (bsz, cfg.world_model_horizon, cfg.obs_dim)
    assert out["state"] is not None
    assert out["state"].mem.shape == (bsz, cfg.memory_slots, cfg.hidden_size)


def test_shapes_no_obs() -> None:
    cfg = _make_cfg()
    model = VAGICore(cfg)
    bsz, tlen = 2, 5
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, tlen), dtype=torch.long)
    state = model.init_state(bsz)

    out = model.forward(input_ids=input_ids, obs=None, state=state)
    seq_len = tlen

    assert out["text_logits"].shape == (bsz, seq_len, cfg.vocab_size)
    assert out["action_logits"].shape == (bsz, cfg.action_dim)
    assert out["value"].shape == (bsz, 1)
    assert out["world_pred"] is not None
    assert out["world_pred"].shape == (bsz, cfg.world_model_horizon, cfg.obs_dim)
    assert out["state"] is not None
    assert out["state"].mem.shape == (bsz, cfg.memory_slots, cfg.hidden_size)
