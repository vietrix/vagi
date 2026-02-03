import torch

from core.base import VAGIConfig, VAGICore


def test_action_validity_filters_plan() -> None:
    cfg = VAGIConfig(
        vocab_size=32,
        hidden_size=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=8,
        obs_tokens=1,
        action_dim=4,
        memory_slots=2,
        dropout=0.0,
        use_world_pred=True,
        world_model_horizon=1,
        use_action_validity=True,
    )
    model = VAGICore(cfg)
    with torch.no_grad():
        model.action_valid.proj.weight.zero_()
        model.action_valid.proj.bias.fill_(-10.0)
        model.action_valid.proj.bias[2] = 10.0

    obs = torch.randn(1, cfg.obs_dim)
    state = model.init_state(1)
    input_ids = torch.zeros((1, 1), dtype=torch.long)

    out = model.plan_step(
        input_ids=input_ids,
        obs=obs,
        state=state,
        num_candidates=4,
        horizon=2,
        action_validity_threshold=0.9,
    )
    assert int(out["action"].item()) == 2
