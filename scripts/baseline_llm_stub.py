"""LLM baseline stub (offline) for CodeEnv."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

from envs.code_env.actions import ACTION_TYPES
from envs.code_env.code_env import CodeEnv
from scripts.baseline_random import action_from_type, _default_patch


class LLMStub:
    """Deterministic fake LLM that maps obs to an action type."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def choose_action_type(self, obs, step: int) -> str:
        score = 0.0
        for idx, value in enumerate(obs.tolist()):
            score += float(value) * (idx + 1)
        score += self.seed * 0.01 + step * 0.1
        if math.isnan(score) or math.isinf(score):
            score = float(step)
        action_idx = int(abs(score) * 1000.0) % len(ACTION_TYPES)
        return ACTION_TYPES[action_idx]


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
    stub = LLMStub(seed=seed)
    total_reward = 0.0
    start = time.perf_counter()
    info: Dict[str, object] = {}
    steps = 0
    patch = _default_patch()
    for step in range(max_steps):
        action_type = stub.choose_action_type(obs, step)
        action = action_from_type(action_type, patch)
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
    parser = argparse.ArgumentParser(description="Offline LLM stub baseline.")
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
