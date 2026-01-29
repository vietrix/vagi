"""Benchmark CodeEnv tasks and report pass rate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import time

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
from envs.code_env.code_env import CodeEnv
from scripts.bench_utils import collect_tasks
from scripts.utils import set_deterministic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CodeEnv tasks.")
    parser.add_argument("--tasks-dir", type=str, default="envs/code_env/fixtures/benchmarks")
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="runs/bench_code_env.json")
    parser.add_argument("--baseline", type=str, default="scripted", choices=["scripted"])
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _load_patch(task_dir: Path) -> str:
    patch_path = task_dir / "solution.patch"
    if not patch_path.exists():
        raise FileNotFoundError(f"Missing solution.patch in {task_dir}")
    return patch_path.read_text(encoding="utf-8")


def run_benchmark(
    tasks_dir: Path,
    obs_dim: int,
    max_steps: int,
    seed: int,
    level: int | None,
    baseline: str,
) -> Dict[str, object]:
    results: List[Dict[str, object]] = []
    passed = 0
    total_steps = 0
    total_runs = 0
    total_time = 0.0
    task_dirs = collect_tasks(tasks_dir, level=level)
    for task_dir in task_dirs:
        env = CodeEnv(obs_dim=obs_dim, max_steps=max_steps, max_run_tests=2, seed=seed, repo_path=task_dir)
        env.reset()

        start = time.perf_counter()
        steps = 0
        runs = 0

        env.step(serialize_action(PlanReadErrorsAction()))
        steps += 1
        env.step(serialize_action(PlanLocateSourceAction()))
        steps += 1
        env.step(serialize_action(ReadFileAction(path="src/buggy.py")))
        steps += 1
        env.step(serialize_action(PlanPatchAction()))
        steps += 1
        patch = _load_patch(task_dir)
        env.step(serialize_action(ApplyPatchAction(path="src/buggy.py", diff=patch)))
        steps += 1
        env.step(serialize_action(PlanVerifyAction()))
        steps += 1
        _obs, _reward, _done, info = env.step(serialize_action(RunTestsAction()))
        steps += 1
        runs += 1
        total_time += time.perf_counter() - start

        success = info.get("fail_count", 1) == 0
        if success:
            passed += 1
        total_steps += steps
        total_runs += runs
        results.append({"task": task_dir.name, "passed": bool(success), "steps": steps, "run_tests": runs})

    total = len(task_dirs)
    pass_rate = passed / total if total else 0.0
    return {
        "baseline": baseline,
        "total": total,
        "passed": passed,
        "pass_rate": pass_rate,
        "avg_steps": total_steps / total if total else 0.0,
        "avg_runs": total_runs / total if total else 0.0,
        "avg_time": total_time / total if total else 0.0,
        "results": results,
    }


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed, args.deterministic)
    tasks_dir = Path(args.tasks_dir)
    report = run_benchmark(tasks_dir, args.obs_dim, args.max_steps, args.seed, args.level, args.baseline)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"pass_rate={report['pass_rate']:.2%} ({report['passed']}/{report['total']})")


if __name__ == "__main__":
    main()
