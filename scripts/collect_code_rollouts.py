"""Collect rollouts from the code environment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from envs.code_env.actions import (
    ApplyPatchAction,
    PlanLocateSourceAction,
    PlanPatchAction,
    PlanReadErrorsAction,
    PlanVerifyAction,
    ReadFileAction,
    RunTestsAction,
    serialize_action,
)
from envs.code_env.code_env import CodeEnv, PATCH_SEPARATOR
from runtime.logging import JsonlWriter
from runtime.privacy import apply_retention, delete_logs
from scripts.utils import set_deterministic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect rollouts from CodeEnv.")
    parser.add_argument("--out", type=str, default="logs/code_rollouts.jsonl")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--policy", type=str, default="scripted", choices=["scripted"])
    parser.add_argument("--privacy-opt-in", action="store_true")
    parser.add_argument("--retain-days", type=int, default=7)
    parser.add_argument("--delete-logs", action="store_true")
    return parser.parse_args()


def _fix_patch() -> str:
    old = "def add(a, b):\n    return a - b"
    new = "def add(a, b):\n    return a + b"
    return old + PATCH_SEPARATOR + new


def _to_list(x: torch.Tensor) -> List[float]:
    return [float(v) for v in x.detach().cpu().tolist()]


def collect_rollouts(
    *,
    out_path: str | Path,
    episodes: int,
    max_steps: int,
    obs_dim: int,
    seed: int,
    deterministic: bool = False,
    privacy_opt_in: bool = False,
    retain_days: int | None = 7,
    delete_existing: bool = False,
) -> List[Dict[str, object]]:
    if episodes <= 0:
        raise ValueError("episodes must be > 0")
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    set_deterministic(seed, deterministic)
    env = CodeEnv(obs_dim=obs_dim, max_steps=max_steps, seed=seed)
    records: List[Dict[str, object]] = []
    patch = _fix_patch()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if delete_existing:
        delete_logs(out_path.parent)
    apply_retention(out_path.parent, retain_days)
    writer = JsonlWriter(out_path, scrub_pii=True, privacy_opt_in=privacy_opt_in)

    try:
        for _ in range(episodes):
            obs = env.reset()
            for step_idx in range(max_steps):
                if step_idx == 0:
                    action = serialize_action(PlanReadErrorsAction())
                elif step_idx == 1:
                    action = serialize_action(PlanLocateSourceAction())
                elif step_idx == 2:
                    action = serialize_action(ReadFileAction(path="src/buggy.py"))
                elif step_idx == 3:
                    action = serialize_action(PlanPatchAction())
                elif step_idx == 4:
                    action = serialize_action(ApplyPatchAction(path="src/buggy.py", diff=patch))
                elif step_idx == 5:
                    action = serialize_action(PlanVerifyAction())
                else:
                    action = serialize_action(RunTestsAction())
                obs_next, reward, done, info = env.step(action)
                record = {
                    "obs": _to_list(obs),
                    "action": action,
                    "reward": float(reward),
                    "done": bool(done),
                    "value": 0.0,
                    "obs_next": _to_list(obs_next),
                    "timestep": int(step_idx),
                    "seed": int(seed),
                    "fail_count": int(info.get("fail_count", 0)),
                    "failing_tests": info.get("failing_tests", []),
                    "top_error_type": info.get("top_error_type", ""),
                }
                records.append(record)
                writer.write(record)
                obs = obs_next
                if done:
                    break
    finally:
        writer.close()
    return records


def main() -> None:
    args = parse_args()
    collect_rollouts(
        out_path=args.out,
        episodes=args.episodes,
        max_steps=args.max_steps,
        obs_dim=args.obs_dim,
        seed=args.seed,
        deterministic=args.deterministic,
        privacy_opt_in=args.privacy_opt_in,
        retain_days=args.retain_days,
        delete_existing=args.delete_logs,
    )
    print(f"Saved code rollouts to {args.out}")


if __name__ == "__main__":
    main()
