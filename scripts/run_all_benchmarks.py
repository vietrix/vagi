"""Run vAGI and baseline agents across tasks and seeds."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch

from envs.code_env.actions import ACTION_DIM, ACTION_TYPES
from envs.code_env.code_env import CodeEnv, PATCH_SEPARATOR
from runtime.privacy import scrub_record
from scripts.baseline_heuristic import run_episode as run_heuristic_episode
from scripts.baseline_random import action_from_type as action_from_type_random
from scripts.baseline_random import run_episode as run_random_episode
from scripts.bench_utils import collect_tasks
from scripts.utils import set_deterministic
from vagi_core import VAGIConfig, VAGICore


@dataclass
class EpisodeRecord:
    task: str
    seed: int
    agent: str
    success: bool
    steps: int
    total_reward: float
    latency_s: float


def _default_patch() -> str:
    old = "def add(a, b):\n    return a - b"
    new = "def add(a, b):\n    return a + b"
    return old + PATCH_SEPARATOR + new


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


def _collect_tasks(tasks_dir: Path, level: int | None, limit: int | None) -> List[Path]:
    return collect_tasks(tasks_dir, level=level, limit=limit)


def _git_commit(repo_root: Path) -> str:
    head_path = repo_root / ".git" / "HEAD"
    if not head_path.exists():
        return "unknown"
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref:"):
        ref = head.split(" ", 1)[1].strip()
        ref_path = repo_root / ".git" / ref
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
        packed = repo_root / ".git" / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or " " not in line:
                    continue
                sha, ref_name = line.strip().split(" ", 1)
                if ref_name == ref:
                    return sha
    return head if head else "unknown"


def _system_info(repo_root: Path) -> Dict[str, object]:
    gpus = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            gpus.append(torch.cuda.get_device_name(idx))
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os_cpu_count(),
        "cpu_name": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpus": gpus,
        "git_commit": _git_commit(repo_root),
    }


def os_cpu_count() -> int:
    count = os.cpu_count() if hasattr(os, "cpu_count") else None
    return int(count) if count is not None else 0


def _aggregate(records: List[EpisodeRecord]) -> Dict[str, float]:
    total = len(records)
    if total == 0:
        return {"pass_rate": 0.0, "mean_reward": 0.0, "mean_steps": 0.0, "mean_latency_s": 0.0}
    pass_rate = sum(1 for r in records if r.success) / total
    mean_reward = sum(r.total_reward for r in records) / total
    mean_steps = sum(r.steps for r in records) / total
    mean_latency = sum(r.latency_s for r in records) / total
    return {
        "pass_rate": pass_rate,
        "mean_reward": mean_reward,
        "mean_steps": mean_steps,
        "mean_latency_s": mean_latency,
    }


def _write_partial(run_dir: Path, config: Dict[str, object], records: List[EpisodeRecord]) -> None:
    by_agent: Dict[str, List[EpisodeRecord]] = {}
    for record in records:
        by_agent.setdefault(record.agent, []).append(record)
    summary = {agent: _aggregate(recs) for agent, recs in by_agent.items()}
    payload = {
        "config": config,
        "summary": summary,
        "records": [asdict(record) for record in records],
    }
    payload = scrub_record(payload)
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_vagi_episode(
    *,
    model: VAGICore,
    task_dir: Path,
    obs_dim: int,
    max_steps: int,
    max_run_tests: int,
    seed: int,
    patch: str,
) -> EpisodeRecord:
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
    return EpisodeRecord(
        task=task_dir.name,
        seed=seed,
        agent="vagi",
        success=success,
        steps=steps,
        total_reward=total_reward,
        latency_s=latency,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vAGI + baselines across tasks and seeds.")
    parser.add_argument("--tasks-dir", type=str, default="envs/code_env/fixtures/benchmarks")
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-run-tests", type=int, default=3)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated list of seeds.")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic(0, args.deterministic)

    tasks_dir = Path(args.tasks_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds)
    tasks = _collect_tasks(tasks_dir, args.level, args.limit_tasks)
    if not tasks:
        raise ValueError("No tasks found.")

    torch.manual_seed(0)
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
    records: List[EpisodeRecord] = []
    config = {
        "tasks_dir": str(tasks_dir),
        "obs_dim": args.obs_dim,
        "max_steps": args.max_steps,
        "max_run_tests": args.max_run_tests,
        "episodes_per_task": args.episodes_per_task,
        "seeds": seeds,
        "level": args.level,
        "limit_tasks": args.limit_tasks,
        "deterministic": bool(args.deterministic),
        "run_dir": str(run_dir),
    }
    info = _system_info(repo_root=Path(__file__).resolve().parents[1])
    (run_dir / "system_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    csv_path = run_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "seed", "agent", "success", "steps", "reward", "latency_s"])

        for seed in seeds:
            set_deterministic(seed, args.deterministic)
            for task_dir in tasks:
                for episode_idx in range(args.episodes_per_task):
                    episode_seed = seed + episode_idx
                    vagi_record = _run_vagi_episode(
                        model=model,
                        task_dir=task_dir,
                        obs_dim=args.obs_dim,
                        max_steps=args.max_steps,
                        max_run_tests=args.max_run_tests,
                        seed=episode_seed,
                        patch=patch,
                    )
                    records.append(vagi_record)
                    writer.writerow(
                        [
                            vagi_record.task,
                            vagi_record.seed,
                            vagi_record.agent,
                            int(vagi_record.success),
                            vagi_record.steps,
                            f"{vagi_record.total_reward:.6f}",
                            f"{vagi_record.latency_s:.6f}",
                        ]
                    )

                    rand = run_random_episode(
                        task_dir=task_dir,
                        obs_dim=args.obs_dim,
                        max_steps=args.max_steps,
                        max_run_tests=args.max_run_tests,
                        seed=episode_seed,
                    )
                    rand_record = EpisodeRecord(
                        task=rand["task"],
                        seed=episode_seed,
                        agent="random",
                        success=bool(rand["success"]),
                        steps=int(rand["steps"]),
                        total_reward=float(rand["total_reward"]),
                        latency_s=float(rand["latency_s"]),
                    )
                    records.append(rand_record)
                    writer.writerow(
                        [
                            rand_record.task,
                            rand_record.seed,
                            rand_record.agent,
                            int(rand_record.success),
                            rand_record.steps,
                            f"{rand_record.total_reward:.6f}",
                            f"{rand_record.latency_s:.6f}",
                        ]
                    )

                    heur = run_heuristic_episode(
                        task_dir=task_dir,
                        obs_dim=args.obs_dim,
                        max_steps=args.max_steps,
                        max_run_tests=args.max_run_tests,
                        seed=episode_seed,
                    )
                    heur_record = EpisodeRecord(
                        task=heur["task"],
                        seed=episode_seed,
                        agent="heuristic",
                        success=bool(heur["success"]),
                        steps=int(heur["steps"]),
                        total_reward=float(heur["total_reward"]),
                        latency_s=float(heur["latency_s"]),
                    )
                    records.append(heur_record)
                    writer.writerow(
                        [
                            heur_record.task,
                            heur_record.seed,
                            heur_record.agent,
                            int(heur_record.success),
                            heur_record.steps,
                            f"{heur_record.total_reward:.6f}",
                            f"{heur_record.latency_s:.6f}",
                        ]
                    )

                _write_partial(run_dir=run_dir, config=config, records=records)
    print(f"Saved benchmark run to {run_dir}")


if __name__ == "__main__":
    main()
