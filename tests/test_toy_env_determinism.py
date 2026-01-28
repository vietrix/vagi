import torch

from envs.toy_env import ToyEnv


def test_toy_env_determinism() -> None:
    env1 = ToyEnv(obs_dim=8, action_dim=4, max_steps=5, seed=123)
    env2 = ToyEnv(obs_dim=8, action_dim=4, max_steps=5, seed=123)

    obs1 = env1.reset()
    obs2 = env2.reset()
    assert torch.allclose(obs1, obs2)

    actions = [0, 1, 2, 3, 0]
    for action in actions:
        next1, reward1, done1, info1 = env1.step(action)
        next2, reward2, done2, info2 = env2.step(action)
        assert torch.allclose(next1, next2)
        assert reward1 == reward2
        assert done1 == done2
        assert info1["target"] == info2["target"]
