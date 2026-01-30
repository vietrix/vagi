import torch

from vagi_core import VAGIConfig, VAGICore


def test_step_consistency() -> None:
    cfg = VAGIConfig(
        vocab_size=128,
        hidden_size=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=16,
        obs_dim=16,
        obs_tokens=2,
        action_dim=8,
        memory_slots=4,
        dropout=0.0,
        use_world_pred=False,
    )
    model = VAGICore(cfg)
    bsz = 2
    obs = torch.randn(bsz, cfg.obs_dim)
    token = torch.randint(0, cfg.vocab_size, (bsz, 1), dtype=torch.long)

    state = model.init_state(bsz)
    for step_idx in range(5):
        out = model.step(input_ids=token, obs=obs, state=state)
        assert set(out.keys()) == {
            "text_logits",
            "action_logits",
            "value",
            "value_logvar",
            "world_pred",
            "world_logvar",
            "error_logits",
            "info_gain",
            "budget_mode_logits",
            "budget_horizon_logits",
            "budget_candidate_logits",
            "confidence",
            "uncertainty",
            "budget",
            "stopReason",
            "state",
        }
        assert out["state"] is not None
        assert out["state"].timestep == step_idx + 1
        assert out["state"].mem.shape == (bsz, cfg.memory_slots, cfg.hidden_size)
        state = out["state"]
