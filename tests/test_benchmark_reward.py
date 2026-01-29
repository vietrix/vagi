from scripts.ablate_memory import run_ablation


def test_reward_regression_guard() -> None:
    results = run_ablation(
        episodes=1,
        episode_length=5,
        seed=0,
        vocab_size=32,
        hidden_size=32,
        layers=1,
        heads=4,
        obs_dim=8,
        obs_tokens=1,
        action_dim=4,
        memory_slots=2,
        target=2,
        policy="heuristic",
        use_special_tokens=True,
    )
    for entry in results:
        assert entry["avg_reward"] >= 0.9
