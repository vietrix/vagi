"""Run a multi-environment curriculum with automatic level progression."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from scripts.collect_multi_env_rollouts import collect_rollouts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-env curriculum progression.")
    parser.add_argument("--tasks-dir", type=str, default="envs/code_env/fixtures/benchmarks")
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-run-tests", type=int, default=2)
    parser.add_argument("--episodes-per-env", type=int, default=5)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--start-level", type=int, default=1)
    parser.add_argument("--max-level", type=int, default=3)
    parser.add_argument("--pass-threshold", type=float, default=0.6)
    parser.add_argument("--toy-action-base", type=int, default=4)
    parser.add_argument("--toy-action-step", type=int, default=2)
    parser.add_argument("--ui-size-base", type=int, default=3)
    parser.add_argument("--ui-size-step", type=int, default=1)
    parser.add_argument("--ui-channels", type=int, default=1)
    parser.add_argument("--policy", type=str, default="model", choices=["model", "random"])
    parser.add_argument("--mode", type=str, default="act", choices=["act", "think"])
    parser.add_argument("--save-rollouts", action="store_true")
    parser.add_argument("--out-dir", type=str, default="results/curriculum_multi_env")
    return parser.parse_args()


def _level_config(level: int, args: argparse.Namespace) -> Dict[str, int]:
    return {
        "toy_action_dim": args.toy_action_base + (level - 1) * args.toy_action_step,
        "ui_image_size": args.ui_size_base + (level - 1) * args.ui_size_step,
        "code_level": level,
    }


def _weighted_pass(metrics: Dict[str, Dict[str, float]]) -> float:
    total_weight = 0.0
    score = 0.0
    for env, env_metrics in metrics.items():
        weight = max(env_metrics.get("mean_steps", 1.0), 1.0)
        total_weight += weight
        score += env_metrics.get("pass_rate", 0.0) * weight
    if total_weight == 0.0:
        return 0.0
    return score / total_weight


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    levels_summary: List[Dict[str, object]] = []
    stop_level = args.max_level

    for level in range(args.start_level, args.max_level + 1):
        level_cfg = _level_config(level, args)
        rollouts_path = run_dir / f"level_{level}.jsonl" if args.save_rollouts else None
        _, metrics = collect_rollouts(
            out_path=rollouts_path,
            tasks_dir=Path(args.tasks_dir),
            level=level_cfg["code_level"],
            episodes_per_env=args.episodes_per_env,
            episodes_per_task=args.episodes_per_task,
            max_steps=args.max_steps,
            max_run_tests=args.max_run_tests,
            obs_dim=args.obs_dim,
            toy_action_dim=level_cfg["toy_action_dim"],
            ui_image_size=level_cfg["ui_image_size"],
            ui_channels=args.ui_channels,
            policy=args.policy,
            mode=args.mode,
            seed=args.seed + level,
            deterministic=args.deterministic,
        )
        combined = _weighted_pass(metrics)
        passed = combined >= args.pass_threshold
        levels_summary.append(
            {
                "level": level,
                "passed": passed,
                "config": level_cfg,
                "metrics": metrics,
                "combined_pass_rate": combined,
            }
        )
        if not passed:
            stop_level = level
            break

    payload = {
        "config": {
            "tasks_dir": args.tasks_dir,
            "obs_dim": args.obs_dim,
            "max_steps": args.max_steps,
            "max_run_tests": args.max_run_tests,
            "episodes_per_env": args.episodes_per_env,
            "episodes_per_task": args.episodes_per_task,
            "start_level": args.start_level,
            "max_level": args.max_level,
            "pass_threshold": args.pass_threshold,
            "policy": args.policy,
            "mode": args.mode,
            "deterministic": bool(args.deterministic),
        },
        "stop_level": stop_level,
        "levels": levels_summary,
    }
    (run_dir / "curriculum.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = run_dir / "curriculum.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "level",
                "env",
                "pass_rate",
                "mean_reward",
                "mean_steps",
                "combined_pass_rate",
            ]
        )
        for entry in levels_summary:
            metrics = entry["metrics"]
            for env, env_metrics in metrics.items():
                writer.writerow(
                    [
                        entry["level"],
                        env,
                        f"{env_metrics['pass_rate']:.6f}",
                        f"{env_metrics['mean_reward']:.6f}",
                        f"{env_metrics['mean_steps']:.6f}",
                        f"{entry['combined_pass_rate']:.6f}",
                    ]
                )
    print(f"Saved curriculum results to {run_dir}")


if __name__ == "__main__":
    main()
