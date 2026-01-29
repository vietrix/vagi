import torch

from vagi_core import VAGIConfig, VAGICore


def test_plan_step_determinism_with_seed() -> None:
    torch.manual_seed(0)
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
        world_model_horizon=2,
        use_uncertainty=True,
    )
    model = VAGICore(cfg)
    model.eval()

    obs = torch.randn(1, cfg.obs_dim)
    state = model.init_state(1)
    input_ids = torch.zeros((1, 1), dtype=torch.long)

    torch.manual_seed(123)
    action_a = model.plan_step(
        input_ids=input_ids,
        obs=obs,
        state=state.clone(),
        num_candidates=4,
        horizon=2,
        uncertainty_weight=1.0,
    )["action"].item()

    torch.manual_seed(123)
    action_b = model.plan_step(
        input_ids=input_ids,
        obs=obs,
        state=state.clone(),
        num_candidates=4,
        horizon=2,
        uncertainty_weight=1.0,
    )["action"].item()

    assert action_a == action_b
