"""Benchmark CodeEnv tasks and report pass rate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from envs.code_env.actions import ApplyPatchAction, ReadFileAction, RunTestsAction, serialize_action
from envs.code_env.code_env import CodeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CodeEnv tasks.")
    parser.add_argument("--tasks-dir", type=str, default="envs/code_env/fixtures/benchmarks")
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="runs/bench_code_env.json")
    return parser.parse_args()


def _load_patch(task_dir: Path) -> str:
    patch_path = task_dir / "solution.patch"
    if not patch_path.exists():
        raise FileNotFoundError(f"Missing solution.patch in {task_dir}")
    return patch_path.read_text(encoding="utf-8")


def run_benchmark(tasks_dir: Path, obs_dim: int, max_steps: int, seed: int) -> Dict[str, object]:
    results: List[Dict[str, object]] = []
    passed = 0
    task_dirs = sorted([p for p in tasks_dir.iterdir() if p.is_dir()])
    for task_dir in task_dirs:
        env = CodeEnv(obs_dim=obs_dim, max_steps=max_steps, max_run_tests=2, seed=seed, repo_path=task_dir)
        env.reset()

        read_action = serialize_action(ReadFileAction(path="src/buggy.py"))
        env.step(read_action)

        patch = _load_patch(task_dir)
        patch_action = serialize_action(ApplyPatchAction(path="src/buggy.py", diff=patch))
        env.step(patch_action)

        run_action = serialize_action(RunTestsAction())
        _obs, _reward, _done, info = env.step(run_action)

        success = info.get("fail_count", 1) == 0
        if success:
            passed += 1
        results.append({"task": task_dir.name, "passed": bool(success)})

    total = len(task_dirs)
    pass_rate = passed / total if total else 0.0
    return {"total": total, "passed": passed, "pass_rate": pass_rate, "results": results}


def main() -> None:
    args = parse_args()
    tasks_dir = Path(args.tasks_dir)
    report = run_benchmark(tasks_dir, args.obs_dim, args.max_steps, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"pass_rate={report['pass_rate']:.2%} ({report['passed']}/{report['total']})")


if __name__ == "__main__":
    main()
