import torch

from envs.code_env.actions import EditAction, RunTestsAction, serialize_action
from envs.code_env.code_env import CodeEnv, PATCH_SEPARATOR


def _fix_patch() -> str:
    old = "def add(a, b):\n    return a - b"
    new = "def add(a, b):\n    return a + b"
    return old + PATCH_SEPARATOR + new


def test_code_env_fix_bug() -> None:
    env = CodeEnv(obs_dim=32, max_steps=3, seed=0)
    obs = env.reset()
    assert isinstance(obs, torch.Tensor)

    edit_action = serialize_action(EditAction(path="src/buggy.py", patch=_fix_patch()))
    obs, reward, done, info = env.step(edit_action)
    assert info["action_id"] == 1
    assert done is False

    run_action = serialize_action(RunTestsAction())
    obs, reward, done, info = env.step(run_action)
    assert isinstance(obs, torch.Tensor)
    assert done is True
    assert reward > 9.0
