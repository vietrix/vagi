from envs.code_env.actions import ApplyPatchAction, ReadFileAction, RunTestsAction, serialize_action
from envs.code_env.code_env import CodeEnv, PATCH_SEPARATOR


def _small_patch() -> str:
    old = "def add(a, b):\n    return a - b"
    new = "def add(a, b):\n    return a + b"
    return old + PATCH_SEPARATOR + new


def test_read_before_write_guardrail() -> None:
    env = CodeEnv(obs_dim=16, max_steps=2, max_patch_chars=500, seed=0)
    env.reset()
    patch_action = serialize_action(ApplyPatchAction(path="src/buggy.py", diff=_small_patch()))
    _obs, _reward, _done, info = env.step(patch_action)
    assert info["patch_applied"] is False
    assert info["tool_ok"] is False


def test_patch_size_guardrail() -> None:
    env = CodeEnv(obs_dim=16, max_steps=2, max_patch_chars=10, seed=0)
    env.reset()
    read_action = serialize_action(ReadFileAction(path="src/buggy.py"))
    env.step(read_action)
    patch_action = serialize_action(ApplyPatchAction(path="src/buggy.py", diff=_small_patch()))
    _obs, _reward, _done, info = env.step(patch_action)
    assert info["patch_applied"] is False
    assert info["tool_ok"] is False


def test_run_tests_limit_guardrail() -> None:
    env = CodeEnv(obs_dim=16, max_steps=3, max_run_tests=1, seed=0)
    env.reset()
    run_action = serialize_action(RunTestsAction())
    env.step(run_action)
    _obs, _reward, done, info = env.step(run_action)
    assert done is True
    assert info["tool_ok"] is False
