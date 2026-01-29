import math

from scripts.rl_a2c_toy import run_a2c


def test_rl_a2c_smoke(tmp_path) -> None:
    log_path = tmp_path / "rl_a2c_toy.jsonl"
    logs = run_a2c(
        episodes=2,
        episode_length=4,
        gamma=0.9,
        lr=1e-3,
        seed=123,
        log_path=log_path,
        vocab_size=32,
        hidden_size=32,
        layers=1,
        heads=4,
        obs_dim=8,
        obs_tokens=1,
        action_dim=4,
        memory_slots=2,
        target=3,
    )

    assert len(logs) == 2
    assert log_path.exists()

    for entry in logs:
        assert math.isfinite(entry["policy_loss"])
        assert math.isfinite(entry["value_loss"])
        assert math.isfinite(entry["total_reward"])
        assert math.isfinite(entry["mean_value"])
