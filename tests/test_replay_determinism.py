from __future__ import annotations

from scripts.baseline_random import run_episode


def test_replay_determinism() -> None:
    task_dir = "envs/code_env/fixtures/mini_repo"
    kwargs = dict(task_dir=task_dir, obs_dim=64, max_steps=4, max_run_tests=2, seed=123)
    first = run_episode(**kwargs)
    second = run_episode(**kwargs)
    assert first["success"] == second["success"]
    assert first["steps"] == second["steps"]
    assert first["total_reward"] == second["total_reward"]
