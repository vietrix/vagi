import torch

from core.base import VAGIConfig, VAGICore


def test_forward_return_features() -> None:
    cfg = VAGIConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=16,
        obs_dim=8,
        obs_tokens=1,
        action_dim=4,
        memory_slots=2,
        dropout=0.0,
        use_world_pred=False,
    )
    model = VAGICore(cfg)
    bsz, tlen = 2, 4
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, tlen), dtype=torch.long)
    obs = torch.randn(bsz, cfg.obs_dim)
    state = model.init_state(bsz)

    out = model.forward(input_ids=input_ids, obs=obs, state=state, return_features=True)
    assert "features" in out
    features = out["features"]

    # Check batch and hidden size (seq_len may vary based on special tokens added)
    assert features["hidden"].shape[0] == bsz
    assert features["hidden"].shape[2] == cfg.hidden_size
    # seq_len should be at least tlen + obs_tokens
    assert features["hidden"].shape[1] >= tlen + cfg.obs_tokens
    assert features["h_last"].shape == (bsz, cfg.hidden_size)
    assert features["h_act"] is not None
    assert features["h_act"].shape == (bsz, cfg.hidden_size)
    assert features["mem_next"] is not None
    assert features["mem_next"].shape == (bsz, cfg.memory_slots, cfg.hidden_size)
