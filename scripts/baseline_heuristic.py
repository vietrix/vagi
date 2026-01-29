"""Deterministic heuristic baseline for CodeEnv."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from envs.code_env.actions import (
    ListDirAction,
    PlanLocateSourceAction,
    PlanReadErrorsAction,
    PlanVerifyAction,
    ReadFileAction,
    RunTestsAction,
    SearchAction,
    serialize_action,
)
from envs.code_env.code_env import CodeEnv


def _choose_action(obs) -> str:
    """Heuristic rule using normalized obs features.

    obs[2] is the normalized step progress, obs[0] is normalized fail count.
    The heuristic cycles through planning and inspection before verifying.
    """
    step_ratio = float(obs[2]) if len(obs) > 2 else 0.0
    fail_ratio = float(obs[0]) if len(obs) > 0 else 0.0
    if step_ratio < 0.25:
        return serialize_action(PlanReadErrorsAction())
    if step_ratio < 0.5:
        return serialize_action(PlanLocateSourceAction())
    if step_ratio < 0.75 and fail_ratio > 0.0:
        return serialize_action(ReadFileAction(path="src/buggy.py"))
    if step_ratio < 0.85:
        return serialize_action(SearchAction(pattern="def "))
    if step_ratio < 0.95:
        return serialize_action(PlanVerifyAction())
    return serialize_action(RunTestsAction())


def run_episode(
    *,
    task_dir: str | Path,
    obs_dim: int,
    max_steps: int,
    max_run_tests: int,
    seed: int,
) -> Dict[str, object]:
    env = CodeEnv(
        obs_dim=obs_dim,
        max_steps=max_steps,
        max_run_tests=max_run_tests,
        seed=seed,
        repo_path=task_dir,
    )
    obs = env.reset()
    total_reward = 0.0
    start = time.perf_counter()
    info: Dict[str, object] = {}
    steps = 0
    for step in range(max_steps):
        action = _choose_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps = step + 1
        if done:
            break
    latency = time.perf_counter() - start
    success = int(info.get("fail_count", 1)) == 0
    return {
        "total_reward": total_reward,
        "steps": steps,
        "success": success,
        "latency_s": latency,
        "task": Path(task_dir).name,
    }


def run_baseline(
    *,
    task_dir: str | Path,
    obs_dim: int,
    max_steps: int,
    max_run_tests: int,
    episodes: int,
    seed: int,
) -> List[Dict[str, object]]:
    results = []
    for idx in range(episodes):
        results.append(
            run_episode(
                task_dir=task_dir,
                obs_dim=obs_dim,
                max_steps=max_steps,
                max_run_tests=max_run_tests,
                seed=seed + idx,
            )
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heuristic baseline for CodeEnv.")
    parser.add_argument("--task", type=str, default="envs/code_env/fixtures/mini_repo")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-run-tests", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_baseline(
        task_dir=args.task,
        obs_dim=args.obs_dim,
        max_steps=args.max_steps,
        max_run_tests=args.max_run_tests,
        episodes=args.episodes,
        seed=args.seed,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for record in results:
        print(json.dumps(record))


if __name__ == "__main__":
    main()
