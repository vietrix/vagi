import json

from envs.toy_env import ToyEnv
from runtime.agent_loop import run_episode
from vagi_core import VAGIConfig, VAGICore


def test_agent_loop_runs(tmp_path) -> None:
    log_path = tmp_path / "transitions.jsonl"
    env = ToyEnv(obs_dim=8, action_dim=4, max_steps=5, seed=0)
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

    steps = run_episode(model, env, steps=3, log_path=str(log_path))
    assert steps >= 1
    assert log_path.exists()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 1
    record = json.loads(lines[0])
    for key in ["timestep", "obs", "action", "reward", "value"]:
        assert key in record
