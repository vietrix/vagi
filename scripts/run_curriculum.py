"""Run curriculum levels until pass-rate threshold is reached."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from runtime.privacy import scrub_record
from scripts.ablation_utils import aggregate, run_vagi_records
from scripts.bench_utils import collect_tasks


def _parse_seeds(text: str | None) -> List[int]:
    if not text:
        return list(range(5))
    return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run curriculum progression.")
    parser.add_argument("--tasks-dir", type=str, default="envs/code_env/fixtures/benchmarks")
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-run-tests", type=int, default=3)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--start-level", type=int, default=1)
    parser.add_argument("--max-level", type=int, default=3)
    parser.add_argument("--pass-threshold", type=float, default=0.6)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--out-dir", type=str, default="results/curriculum")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    tasks_dir = Path(args.tasks_dir)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    levels = []
    stop_level = args.max_level
    for level in range(args.start_level, args.max_level + 1):
        tasks = collect_tasks(tasks_dir, level=level)
        if not tasks:
            stop_level = level - 1
            break
        records = run_vagi_records(
            tasks=tasks,
            seeds=seeds,
            episodes_per_task=args.episodes_per_task,
            obs_dim=args.obs_dim,
            max_steps=args.max_steps,
            max_run_tests=args.max_run_tests,
            deterministic=args.deterministic,
            config_overrides={},
            use_kv_cache=True,
        )
        metrics = aggregate(records)
        passed = metrics["pass_rate"] >= args.pass_threshold
        levels.append(
            {
                "level": level,
                "passed": passed,
                "metrics": metrics,
                "episodes": len(records),
            }
        )
        if not passed:
            stop_level = level
            break

    payload = scrub_record(
        {
            "config": {
                "tasks_dir": args.tasks_dir,
                "obs_dim": args.obs_dim,
                "max_steps": args.max_steps,
                "max_run_tests": args.max_run_tests,
                "episodes_per_task": args.episodes_per_task,
                "seeds": seeds,
                "start_level": args.start_level,
                "max_level": args.max_level,
                "pass_threshold": args.pass_threshold,
                "deterministic": bool(args.deterministic),
            },
            "stop_level": stop_level,
            "levels": levels,
        }
    )
    (run_dir / "curriculum.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = run_dir / "curriculum.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "level",
                "passed",
                "pass_rate",
                "mean_reward",
                "mean_steps",
                "mean_latency_s",
                "episodes",
            ]
        )
        for entry in levels:
            metrics = entry["metrics"]
            writer.writerow(
                [
                    entry["level"],
                    int(entry["passed"]),
                    f"{metrics['pass_rate']:.6f}",
                    f"{metrics['mean_reward']:.6f}",
                    f"{metrics['mean_steps']:.6f}",
                    f"{metrics['mean_latency_s']:.6f}",
                    entry["episodes"],
                ]
            )
    print(f"Saved curriculum results to {run_dir}")


if __name__ == "__main__":
    main()
