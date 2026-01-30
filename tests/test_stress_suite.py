import pytest
import torch

from vagi_core import VAGIConfig, VAGICore


def _build_model() -> VAGICore:
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
        use_uncertainty=True,
    )
    return VAGICore(cfg)


def test_ood_refuse_on_high_uncertainty() -> None:
    model = _build_model()
    with torch.no_grad():
        if model.world_logvar is not None:
            model.world_logvar.proj.bias.fill_(5.0)

    obs = torch.zeros((1, model.cfg.obs_dim))
    state = model.init_state(1)
    input_ids = torch.zeros((1, 1), dtype=torch.long)
    out = model.plan_step(
        input_ids=input_ids,
        obs=obs,
        state=state,
        ood_uncertainty_threshold=0.1,
        ood_policy="refuse",
    )
    assert out["stopReason"] == "refuse"
    assert out["mode"] == "act"


def test_min_confidence_triggers_info() -> None:
    model = _build_model()
    obs = torch.zeros((1, model.cfg.obs_dim))
    state = model.init_state(1)
    input_ids = torch.zeros((1, 1), dtype=torch.long)
    out = model.plan_step(
        input_ids=input_ids,
        obs=obs,
        state=state,
        min_confidence_to_act=0.95,
    )
    assert out["stopReason"] == "needsInfo"


def test_plan_step_handles_nan_obs() -> None:
    model = _build_model()
    obs = torch.tensor([[float("nan"), float("inf"), -float("inf"), 1.0, -1.0, 0.0, 2.0, -2.0]])
    state = model.init_state(1)
    input_ids = torch.zeros((1, 1), dtype=torch.long)
    out = model.plan_step(input_ids=input_ids, obs=obs, state=state, policy_only=True)
    assert torch.isfinite(out["action_logits"]).all()


def test_missing_obs_raises() -> None:
    model = _build_model()
    state = model.init_state(1)
    input_ids = torch.zeros((1, 1), dtype=torch.long)
    with pytest.raises(ValueError):
        model.plan_step(input_ids=input_ids, obs=None, state=state)
