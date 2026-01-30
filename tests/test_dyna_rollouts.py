import torch

from vagi_core import VAGIConfig, VAGICore
from vagi_core.dyna import dyna_update, imagine_rollouts, mix_rollouts, policy_value_losses


def _build_model() -> VAGICore:
    cfg = VAGIConfig(
        vocab_size=32,
        hidden_size=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=6,
        obs_tokens=1,
        action_dim=4,
        memory_slots=2,
        dropout=0.0,
        use_world_pred=True,
        world_model_horizon=1,
    )
    return VAGICore(cfg)


def test_imagine_rollouts_shapes() -> None:
    model = _build_model()
    obs = torch.zeros((1, model.cfg.obs_dim))
    state = model.init_state(batch_size=1)
    rollouts = imagine_rollouts(model, obs, state, horizon=3, num_rollouts=2)

    assert rollouts.obs.shape == (2, 3, model.cfg.obs_dim)
    assert rollouts.actions.shape == (2, 3)
    assert rollouts.rewards.shape == (2, 3)
    assert rollouts.dones.shape == (2, 3)
    assert rollouts.values is not None
    assert rollouts.values.shape == (2, 4)


def test_mix_rollouts_and_losses() -> None:
    model = _build_model()
    obs = torch.zeros((1, model.cfg.obs_dim))
    state = model.init_state(batch_size=1)
    rollouts = imagine_rollouts(model, obs, state, horizon=2, num_rollouts=2)
    mixed = mix_rollouts(rollouts, rollouts, imagine_ratio=0.5)
    losses = policy_value_losses(model, mixed, gamma=0.9, lam=0.8)
    assert torch.isfinite(losses["policy_loss"]).all()
    assert torch.isfinite(losses["value_loss"]).all()


def test_dyna_update_step() -> None:
    model = _build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    obs = torch.zeros((1, model.cfg.obs_dim))
    state = model.init_state(batch_size=1)
    real = imagine_rollouts(model, obs, state, horizon=2, num_rollouts=1)
    losses = dyna_update(
        model,
        optimizer,
        real,
        imagine_ratio=1.0,
        imagine_horizon=2,
        num_imagined=1,
        obs=obs,
        state=state,
    )
    assert torch.isfinite(losses["total_loss"]).all()
