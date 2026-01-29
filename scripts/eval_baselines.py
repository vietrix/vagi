"""Evaluate vAGI against random and heuristic baselines."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import torch

from envs.code_env.actions import ACTION_DIM, ACTION_TYPES
from envs.code_env.code_env import CodeEnv, PATCH_SEPARATOR
from scripts.baseline_heuristic import run_episode as run_heuristic_episode
from scripts.baseline_random import action_from_type as action_from_type_random
from scripts.baseline_random import run_episode as run_random_episode
from scripts.utils import set_deterministic
from vagi_core import VAGIConfig, VAGICore


def _default_patch() -> str:
    old = "def add(a, b):\n    return a - b"
    new = "def add(a, b):\n    return a + b"
    return old + PATCH_SEPARATOR + new


def _load_manifest(tasks_dir: Path) -> Dict[str, List[str]]:
    manifest_path = tasks_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return data.get("levels", {})


def _collect_tasks(tasks_dir: Path, level: int | None, task: str | None) -> List[Path]:
    if task is not None:
        return [Path(task)]
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Missing tasks dir: {tasks_dir}")
    task_dirs = sorted([p for p in tasks_dir.iterdir() if p.is_dir()])
    if level is None:
        return task_dirs
    levels = _load_manifest(tasks_dir)
    allowed = set(levels.get(str(level), []))
    return [p for p in task_dirs if p.name in allowed]


def _episode_specs(tasks: List[Path], episodes: int, seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    specs: List[Dict[str, object]] = []
    for idx in range(episodes):
        task = tasks[rng.randrange(len(tasks))]
        specs.append({"episode": idx, "task": task, "seed": seed + idx})
    return specs


def _aggregate(records: List[Dict[str, object]]) -> Dict[str, float]:
    total = len(records)
    if total == 0:
        return {"pass_rate": 0.0, "mean_reward": 0.0, "mean_steps": 0.0, "mean_latency_s": 0.0}
    pass_rate = sum(1 for r in records if r["success"]) / total
    mean_reward = sum(float(r["total_reward"]) for r in records) / total
    mean_steps = sum(float(r["steps"]) for r in records) / total
    mean_latency = sum(float(r["latency_s"]) for r in records) / total
    return {
        "pass_rate": pass_rate,
        "mean_reward": mean_reward,
        "mean_steps": mean_steps,
        "mean_latency_s": mean_latency,
    }


def _run_vagi_episode(
    *,
    model: VAGICore,
    task_dir: Path,
    obs_dim: int,
    max_steps: int,
    max_run_tests: int,
    seed: int,
    patch: str,
) -> Dict[str, object]:
    env = CodeEnv(
        obs_dim=obs_dim,
        max_steps=max_steps,
        max_run_tests=max_run_tests,
        seed=seed,
        repo_path=task_dir,
    )
    obs = env.reset()
    state = model.init_state(batch_size=1)
    total_reward = 0.0
    start = time.perf_counter()
    info: Dict[str, object] = {}
    steps = 0
    for step in range(max_steps):
        input_ids = torch.zeros((1, 1), dtype=torch.long)
        out = model.step(input_ids=input_ids, obs=obs.unsqueeze(0), state=state)
        action_id = int(torch.argmax(out["action_logits"], dim=-1).item()) % ACTION_DIM
        action_type = ACTION_TYPES[action_id]
        action = action_from_type_random(action_type, patch)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps = step + 1
        state = out["state"]
        if done:
            break
    latency = time.perf_counter() - start
    success = int(info.get("fail_count", 1)) == 0
    return {
        "total_reward": total_reward,
        "steps": steps,
        "success": success,
        "latency_s": latency,
        "task": task_dir.name,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate vAGI and baseline agents.")
    parser.add_argument("--tasks-dir", type=str, default="envs/code_env/fixtures/benchmarks")
    parser.add_argument("--task", type=str, default=None, help="Optional single-task path")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-run-tests", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--out", type=str, default="results/baselines.json")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed, args.deterministic)

    tasks = _collect_tasks(Path(args.tasks_dir), args.level, args.task)
    if not tasks:
        raise ValueError("No tasks available for evaluation.")
    specs = _episode_specs(tasks, args.episodes, args.seed)

    torch.manual_seed(args.seed)
    model_cfg = VAGIConfig(
        vocab_size=256,
        hidden_size=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=args.obs_dim,
        obs_tokens=2,
        action_dim=ACTION_DIM,
        memory_slots=4,
        dropout=0.0,
        use_world_pred=False,
    )
    model = VAGICore(model_cfg)
    model.eval()

    patch = _default_patch()
    vagi_records: List[Dict[str, object]] = []
    random_records: List[Dict[str, object]] = []
    heuristic_records: List[Dict[str, object]] = []

    for spec in specs:
        task_dir = Path(spec["task"])
        seed = int(spec["seed"])
        vagi_records.append(
            _run_vagi_episode(
                model=model,
                task_dir=task_dir,
                obs_dim=args.obs_dim,
                max_steps=args.max_steps,
                max_run_tests=args.max_run_tests,
                seed=seed,
                patch=patch,
            )
        )
        random_records.append(
            run_random_episode(
                task_dir=task_dir,
                obs_dim=args.obs_dim,
                max_steps=args.max_steps,
                max_run_tests=args.max_run_tests,
                seed=seed,
            )
        )
        heuristic_records.append(
            run_heuristic_episode(
                task_dir=task_dir,
                obs_dim=args.obs_dim,
                max_steps=args.max_steps,
                max_run_tests=args.max_run_tests,
                seed=seed,
            )
        )

    results = {
        "config": {
            "tasks_dir": args.tasks_dir,
            "task": args.task,
            "episodes": args.episodes,
            "obs_dim": args.obs_dim,
            "max_steps": args.max_steps,
            "max_run_tests": args.max_run_tests,
            "seed": args.seed,
            "level": args.level,
            "deterministic": bool(args.deterministic),
        },
        "episodes": [{"task": Path(spec["task"]).name, "seed": spec["seed"]} for spec in specs],
        "agents": {
            "vagi": {"metrics": _aggregate(vagi_records), "episodes": vagi_records},
            "random": {"metrics": _aggregate(random_records), "episodes": random_records},
            "heuristic": {"metrics": _aggregate(heuristic_records), "episodes": heuristic_records},
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    def _row(name: str, metrics: Dict[str, float]) -> str:
        return (
            f"{name:<10} | pass_rate={metrics['pass_rate']:.2f} | "
            f"mean_reward={metrics['mean_reward']:.3f} | mean_steps={metrics['mean_steps']:.2f} | "
            f"latency_s={metrics['mean_latency_s']:.3f}"
        )

    print(_row("vagi", results["agents"]["vagi"]["metrics"]))
    print(_row("random", results["agents"]["random"]["metrics"]))
    print(_row("heuristic", results["agents"]["heuristic"]["metrics"]))


if __name__ == "__main__":
    main()
