from vagi_core.diagnostics import compute_drop, should_rollback


def test_compute_drop_and_rollback() -> None:
    baseline = {"act_mean_reward": 5.0}
    current = {"act_mean_reward": 4.0}
    drop = compute_drop(baseline, current, key="act_mean_reward")
    assert drop == 1.0
    assert should_rollback(drop, threshold=0.5) is True
