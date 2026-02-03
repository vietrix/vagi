import torch

from core.base import VAGIConfig, VAGICore


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
        uncertainty_obs_scale=1.0,
    )
    model = VAGICore(cfg)
    with torch.no_grad():
        if model.value_logvar is not None:
            model.value_logvar.proj.weight.zero_()
            model.value_logvar.proj.bias.zero_()
        if model.world_logvar is not None:
            model.world_logvar.proj.weight.zero_()
            model.world_logvar.proj.bias.zero_()
    return model


def test_uncertainty_increases_with_noise() -> None:
    model = _make_model()
    input_ids = torch.zeros((1, 1), dtype=torch.long)
    state = model.init_state(batch_size=1)
    base = torch.linspace(-1.0, 1.0, model.cfg.obs_dim)
    obs_low = (base * 0.1).unsqueeze(0)
    obs_high = (base * 1.0).unsqueeze(0)

    out_low = model.forward(input_ids=input_ids, obs=obs_low, state=state)
    out_high = model.forward(input_ids=input_ids, obs=obs_high, state=state)

    assert out_low["value_logvar"] is not None
    assert out_high["value_logvar"] is not None
    assert out_low["world_logvar"] is not None
    assert out_high["world_logvar"] is not None

    assert out_high["value_logvar"].mean() > out_low["value_logvar"].mean()
    assert out_high["world_logvar"].mean() > out_low["world_logvar"].mean()


def test_uncertainty_finite_and_positive_var() -> None:
    model = _make_model()
    input_ids = torch.zeros((1, 1), dtype=torch.long)
    state = model.init_state(batch_size=1)
    obs = torch.linspace(-1.0, 1.0, model.cfg.obs_dim).unsqueeze(0)

    out = model.forward(input_ids=input_ids, obs=obs, state=state)
    value_logvar = out["value_logvar"]
    world_logvar = out["world_logvar"]

    assert value_logvar is not None
    assert world_logvar is not None
    assert torch.isfinite(value_logvar).all()
    assert torch.isfinite(world_logvar).all()

    value_var = torch.exp(value_logvar)
    world_var = torch.exp(world_logvar)
    assert torch.isfinite(value_var).all()
    assert torch.isfinite(world_var).all()
    assert (value_var > 0).all()
    assert (world_var > 0).all()
