"""Ablate KV cache on/off for vAGI."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from runtime.privacy import scrub_record
from scripts.ablation_utils import aggregate, aggregate_by_task, run_vagi_records
from scripts.bench_utils import collect_tasks


def _parse_seeds(text: str | None) -> List[int]:
    if not text:
        return list(range(5))
    return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate KV cache on/off.")
    parser.add_argument("--tasks-dir", type=str, default="envs/code_env/fixtures/benchmarks")
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-run-tests", type=int, default=3)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--out-dir", type=str, default="results/ablations")
    return parser.parse_args()


def _write_csv(path: Path, rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    tasks = collect_tasks(Path(args.tasks_dir), level=args.level, limit=args.limit_tasks)
    seeds = _parse_seeds(args.seeds)

    variants: Dict[str, Dict[str, object]] = {}
    for name, use_kv in [("on", True), ("off", False)]:
        records = run_vagi_records(
            tasks=tasks,
            seeds=seeds,
            episodes_per_task=args.episodes_per_task,
            obs_dim=args.obs_dim,
            max_steps=args.max_steps,
            max_run_tests=args.max_run_tests,
            deterministic=args.deterministic,
            config_overrides={},
            use_kv_cache=use_kv,
        )
        variants[name] = {
            "config": {"use_kv_cache": use_kv},
            "metrics": aggregate(records),
            "per_task": aggregate_by_task(records),
            "records": records,
        }

    payload = scrub_record(
        {
            "ablation": "kv_cache",
            "config": {
                "tasks_dir": args.tasks_dir,
                "obs_dim": args.obs_dim,
                "max_steps": args.max_steps,
                "max_run_tests": args.max_run_tests,
                "episodes_per_task": args.episodes_per_task,
                "seeds": seeds,
                "level": args.level,
                "limit_tasks": args.limit_tasks,
                "deterministic": bool(args.deterministic),
            },
            "variants": variants,
        }
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "kv_cache.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = [
        [
            "variant",
            "task",
            "episodes",
            "pass_rate",
            "mean_reward",
            "mean_steps",
            "mean_latency_s",
        ]
    ]
    for variant, data in variants.items():
        per_task = data["per_task"]
        for task_name in sorted(per_task.keys()):
            metrics = per_task[task_name]
            rows.append(
                [
                    variant,
                    task_name,
                    int(metrics.get("episodes", 0.0)),
                    f"{metrics['pass_rate']:.6f}",
                    f"{metrics['mean_reward']:.6f}",
                    f"{metrics['mean_steps']:.6f}",
                    f"{metrics['mean_latency_s']:.6f}",
                ]
            )
    _write_csv(out_dir / "kv_cache.csv", rows)


if __name__ == "__main__":
    main()
