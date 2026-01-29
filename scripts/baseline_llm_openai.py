"""OpenAI LLM baseline for CodeEnv (optional)."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

from envs.code_env.actions import ACTION_TYPES
from envs.code_env.code_env import CodeEnv
from scripts.baseline_random import action_from_type, _default_patch


def is_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _call_openai(prompt: str, model: str, temperature: float) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the OpenAI baseline.")
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("openai package is required. Install with `pip install openai`.") from exc
    base_url = os.getenv("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
    )
    text = getattr(response, "output_text", None)
    if not text:
        raise RuntimeError("OpenAI response missing output_text")
    return text.strip()


def _select_action_type(obs, step: int, model: str, temperature: float) -> str:
    obs_summary = ", ".join(f"{float(x):.3f}" for x in obs[:8].tolist())
    prompt = (
        "Pick one action type from the list below. Reply with only the action type token.\n"
        f"Actions: {', '.join(ACTION_TYPES)}\n"
        f"Step: {step}\n"
        f"Obs features: [{obs_summary}]\n"
    )
    text = _call_openai(prompt, model=model, temperature=temperature)
    for action in ACTION_TYPES:
        if action in text:
            return action
    return "RUN_TESTS"


def run_episode(
    *,
    task_dir: str | Path,
    obs_dim: int,
    max_steps: int,
    max_run_tests: int,
    seed: int,
    model: str,
    temperature: float,
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
    patch = _default_patch()
    for step in range(max_steps):
        action_type = _select_action_type(obs, step, model=model, temperature=temperature)
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
    model: str,
    temperature: float,
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
                model=model,
                temperature=temperature,
            )
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI LLM baseline.")
    parser.add_argument("--task", type=str, default="envs/code_env/fixtures/mini_repo")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-run-tests", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
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
        model=args.model,
        temperature=args.temperature,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for record in results:
        print(json.dumps(record))


if __name__ == "__main__":
    main()
