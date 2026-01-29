"""Run CodeEnv benchmarks across multiple seeds for reproducibility."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

from scripts.bench_code_env import run_benchmark
from scripts.utils import set_deterministic


def _parse_seeds(text: str | None) -> List[int]:
    if not text:
        return list(range(10))
    seeds: List[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        seeds.append(int(chunk))
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CodeEnv benchmarks across seeds.")
    parser.add_argument("--tasks-dir", type=str, default="envs/code_env/fixtures/benchmarks")
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated list of seeds.")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--baseline", type=str, default="scripted", choices=["scripted"])
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks_dir = Path(args.tasks_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds)
    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tasks_dir": str(tasks_dir),
        "obs_dim": args.obs_dim,
        "max_steps": args.max_steps,
        "baseline": args.baseline,
        "level": args.level,
        "deterministic": bool(args.deterministic),
        "seeds": seeds,
        "results": [],
    }

    for seed in seeds:
        set_deterministic(seed, args.deterministic)
        report = run_benchmark(
            tasks_dir=tasks_dir,
            obs_dim=args.obs_dim,
            max_steps=args.max_steps,
            seed=seed,
            level=args.level,
            baseline=args.baseline,
        )
        report["seed"] = seed
        path = out_dir / f"bench_code_env_seed{seed}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        summary["results"].append(report)

    totals = len(summary["results"])
    summary["avg_pass_rate"] = (
        sum(r["pass_rate"] for r in summary["results"]) / totals if totals else 0.0
    )
    summary["avg_steps"] = sum(r["avg_steps"] for r in summary["results"]) / totals if totals else 0.0
    summary["avg_runs"] = sum(r["avg_runs"] for r in summary["results"]) / totals if totals else 0.0
    summary["avg_time"] = sum(r["avg_time"] for r in summary["results"]) / totals if totals else 0.0

    summary_path = out_dir / "bench_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()
