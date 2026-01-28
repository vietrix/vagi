import torch

from vagi_core import VAGIConfig, VAGICore

from scripts.toy_env import ToyEnv


def test_toy_env_agent_loop() -> None:
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
        use_world_pred=False,
    )
    model = VAGICore(cfg)
    model.eval()

    env = ToyEnv(obs_dim=cfg.obs_dim, action_dim=cfg.action_dim, max_steps=5, target=2)
    obs = env.reset()
    state = model.init_state(batch_size=1)
    token = torch.zeros((1, 1), dtype=torch.long)

    steps = 0
    done = False
    while not done and steps < env.max_steps:
        out = model.step(input_ids=token, obs=obs.unsqueeze(0), state=state)
        action = int(torch.argmax(out["action_logits"], dim=-1).item())
        result = env.step(action)
        state = out["state"]
        obs = result.obs
        token = torch.tensor([[action]], dtype=torch.long)
        done = result.done
        steps += 1

    assert steps >= 1
    assert state is not None
    assert state.timestep == steps
