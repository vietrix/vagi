import torch

from envs.code_env.actions import ApplyPatchAction, ReadFileAction, RunTestsAction, serialize_action
from envs.code_env.code_env import CodeEnv, PATCH_SEPARATOR


def _fix_patch() -> str:
    old = "def add(a, b):\n    return a - b"
    new = "def add(a, b):\n    return a + b"
    return old + PATCH_SEPARATOR + new


def test_code_env_fix_bug() -> None:
    env = CodeEnv(obs_dim=32, max_steps=3, seed=0)
    obs = env.reset()
    assert isinstance(obs, torch.Tensor)

    read_action = serialize_action(ReadFileAction(path="src/buggy.py"))
    obs, reward, done, info = env.step(read_action)
    assert info["action_id"] == 1
    assert done is False

    patch_action = serialize_action(ApplyPatchAction(path="src/buggy.py", diff=_fix_patch()))
    obs, reward, done, info = env.step(patch_action)
    assert info["action_id"] == 4
    assert info["patch_applied"] is True
    assert done is False

    run_action = serialize_action(RunTestsAction())
    obs, reward, done, info = env.step(run_action)
    assert isinstance(obs, torch.Tensor)
    assert done is True
    assert reward > 9.0
    assert "failing_tests" in info
    assert "top_error_type" in info
