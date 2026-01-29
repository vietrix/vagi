import torch

from vagi_core import VAGIConfig, VAGICore


def _make_model() -> VAGICore:
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
        use_reflection=True,
        use_budget_head=True,
        budget_max_horizon=3,
        budget_max_candidates=4,
    )
    model = VAGICore(cfg)
    with torch.no_grad():
        model.budget_head.mode.weight.zero_()
        model.budget_head.mode.bias[:] = torch.tensor([5.0, -5.0])
    return model


def test_think_then_act_not_worse_on_easy_task() -> None:
    torch.manual_seed(0)
    model = _make_model()
    obs = torch.randn(1, model.cfg.obs_dim)
    state = model.init_state(1)
    input_ids = torch.zeros((1, 1), dtype=torch.long)

    act_out = model.act(input_ids=input_ids, obs=obs, state=state)
    think_out = model.think_then_act(input_ids=input_ids, obs=obs, state=state)

    assert int(think_out["action"].item()) == int(act_out["action"].item())


def test_budget_controller_prefers_act() -> None:
    torch.manual_seed(1)
    model = _make_model()
    obs = torch.randn(1, model.cfg.obs_dim)
    state = model.init_state(1)
    input_ids = torch.zeros((1, 1), dtype=torch.long)

    for _ in range(5):
        out = model.think_then_act(input_ids=input_ids, obs=obs, state=state)
        assert out["mode"] == "act"
