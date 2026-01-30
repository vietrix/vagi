"""Cross-environment generalization benchmark for vAGI."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

from envs.code_env.actions import (
    ACTION_DIM,
    ACTION_TYPES,
    ApplyPatchAction,
    ListDirAction,
    PlanLocateSourceAction,
    PlanPatchAction,
    PlanReadErrorsAction,
    PlanVerifyAction,
    ReadFileAction,
    RunTestsAction,
    SearchAction,
    serialize_action,
)
from envs.code_env.code_env import CodeEnv, PATCH_SEPARATOR
from envs.toy_env import ToyEnv
from envs.ui_env import UIEnv
from scripts.bench_utils import collect_tasks
from scripts.utils import set_deterministic
from vagi_core import RecurrentState, VAGIConfig, VAGICore


@dataclass
class EpisodeRecord:
    env: str
    task: str
    seed: int
    agent: str
    success: bool
    steps: int
    total_reward: float
    latency_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-env benchmark for vAGI.")
    parser.add_argument("--tasks-dir", type=str, default="envs/code_env/fixtures/benchmarks")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--toy-action-dim", type=int, default=4)
    parser.add_argument("--ui-image-size", type=int, default=4)
    parser.add_argument("--ui-channels", type=int, default=1)
    parser.add_argument("--mode", type=str, default="act", choices=["act", "think"])
    parser.add_argument("--out", type=str, default="results/cross_env.json")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _parse_seeds(text: str) -> List[int]:
    seeds = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        seeds.append(int(chunk))
    return seeds


def _default_patch() -> str:
    old = "def add(a, b):\n    return a - b"
    new = "def add(a, b):\n    return a + b"
    return old + PATCH_SEPARATOR + new


def _load_patch(task_dir: Path) -> str:
    patch_path = task_dir / "solution.patch"
    if patch_path.exists():
        return patch_path.read_text(encoding="utf-8")
    return _default_patch()


def _code_action_from_id(action_id: int, patch_text: str) -> str:
    action_type = ACTION_TYPES[action_id % len(ACTION_TYPES)]
    if action_type == "RUN_TESTS":
        return serialize_action(RunTestsAction())
    if action_type == "READ_FILE":
        return serialize_action(ReadFileAction(path="src/buggy.py"))
    if action_type == "LIST_DIR":
        return serialize_action(ListDirAction(path="src"))
    if action_type == "SEARCH":
        return serialize_action(SearchAction(pattern="return"))
    if action_type == "APPLY_PATCH":
        return serialize_action(ApplyPatchAction(path="src/buggy.py", diff=patch_text))
    if action_type == "PLAN_READ_ERRORS":
        return serialize_action(PlanReadErrorsAction())
    if action_type == "PLAN_LOCATE_SOURCE":
        return serialize_action(PlanLocateSourceAction())
    if action_type == "PLAN_PATCH":
        return serialize_action(PlanPatchAction())
    if action_type == "PLAN_VERIFY":
        return serialize_action(PlanVerifyAction())
    return serialize_action(PlanReadErrorsAction())


def _select_action(
    model: VAGICore,
    obs: torch.Tensor,
    state: RecurrentState,
    *,
    mode: str,
    use_image: bool,
    task_ids: torch.Tensor | None = None,
) -> Tuple[int, torch.Tensor]:
    input_ids = torch.zeros((1, 1), dtype=torch.long)
    obs_batch = None if use_image else obs.unsqueeze(0)
    image_batch = obs.unsqueeze(0) if use_image else None

    if mode == "think":
        plan = model.think_then_act(
            input_ids=input_ids,
            obs=obs_batch,
            image=image_batch,
            state=state,
            task_ids=task_ids,
        )
        action = int(plan["action"].view(-1)[0].item())
        if "outputs" in plan:
            next_state = plan["outputs"]["state"]
        else:
            out = model.step(
                input_ids=input_ids,
                obs=obs_batch,
                image=image_batch,
                state=state,
                task_ids=task_ids,
            )
            next_state = out["state"]
        return action, next_state

    out = model.act(
        input_ids=input_ids,
        obs=obs_batch,
        image=image_batch,
        state=state,
        task_ids=task_ids,
    )
    action = int(out["action"].view(-1)[0].item())
    return action, out["outputs"]["state"]


def _aggregate(records: Iterable[EpisodeRecord]) -> Dict[str, float]:
    records = list(records)
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


def run_toy(
    model: VAGICore,
    seeds: List[int],
    episodes: int,
    *,
    obs_dim: int,
    action_dim: int,
    max_steps: int,
    mode: str,
    deterministic: bool,
) -> List[EpisodeRecord]:
    records: List[EpisodeRecord] = []
    for seed in seeds:
        set_deterministic(seed, deterministic)
        for ep in range(episodes):
            env = ToyEnv(obs_dim=obs_dim, action_dim=action_dim, max_steps=max_steps, seed=seed + ep)
            obs = env.reset()
            state = model.init_state(batch_size=1)
            total_reward = 0.0
            steps = 0
            start = time.perf_counter()
            done = False
            while not done:
                action, state = _select_action(model, obs, state, mode=mode, use_image=False)
                obs, reward, done, _info = env.step(action % env.action_dim)
                total_reward += float(reward)
                steps += 1
            latency = time.perf_counter() - start
            records.append(
                EpisodeRecord(
                    env="toy",
                    task="toy_env",
                    seed=seed,
                    agent=f"vagi_{mode}",
                    success=total_reward > 0.0,
                    steps=steps,
                    total_reward=total_reward,
                    latency_s=latency,
                )
            )
    return records


def run_ui(
    model: VAGICore,
    seeds: List[int],
    episodes: int,
    *,
    image_size: int,
    action_dim: int,
    max_steps: int,
    channels: int,
    mode: str,
    deterministic: bool,
) -> List[EpisodeRecord]:
    records: List[EpisodeRecord] = []
    for seed in seeds:
        set_deterministic(seed, deterministic)
        for ep in range(episodes):
            env = UIEnv(image_size=image_size, action_dim=action_dim, max_steps=max_steps, seed=seed + ep, channels=channels)
            obs = env.reset()
            state = model.init_state(batch_size=1)
            total_reward = 0.0
            steps = 0
            start = time.perf_counter()
            done = False
            while not done:
                action, state = _select_action(model, obs, state, mode=mode, use_image=True)
                obs, reward, done, _info = env.step(action % env.action_dim)
                total_reward += float(reward)
                steps += 1
            latency = time.perf_counter() - start
            records.append(
                EpisodeRecord(
                    env="ui",
                    task="ui_env",
                    seed=seed,
                    agent=f"vagi_{mode}",
                    success=total_reward > 0.0,
                    steps=steps,
                    total_reward=total_reward,
                    latency_s=latency,
                )
            )
    return records


def run_code_env(
    model: VAGICore,
    seeds: List[int],
    tasks_dir: Path,
    *,
    obs_dim: int,
    max_steps: int,
    mode: str,
    deterministic: bool,
) -> List[EpisodeRecord]:
    records: List[EpisodeRecord] = []
    task_dirs = collect_tasks(tasks_dir)
    for seed in seeds:
        set_deterministic(seed, deterministic)
        for task_dir in task_dirs:
            patch_text = _load_patch(task_dir)
            env = CodeEnv(obs_dim=obs_dim, max_steps=max_steps, max_run_tests=2, seed=seed, repo_path=task_dir)
            obs = env.reset()
            state = model.init_state(batch_size=1)
            total_reward = 0.0
            steps = 0
            start = time.perf_counter()
            done = False
            last_info: Dict[str, object] = {}
            while not done:
                action_id, state = _select_action(model, obs, state, mode=mode, use_image=False)
                action_text = _code_action_from_id(action_id, patch_text)
                obs, reward, done, info = env.step(action_text)
                last_info = info
                total_reward += float(reward)
                steps += 1
            latency = time.perf_counter() - start
            success = bool(last_info.get("fail_count", 1) == 0)
            records.append(
                EpisodeRecord(
                    env="code",
                    task=task_dir.name,
                    seed=seed,
                    agent=f"vagi_{mode}",
                    success=success,
                    steps=steps,
                    total_reward=total_reward,
                    latency_s=latency,
                )
            )
    return records


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    ui_action_dim = args.ui_image_size * args.ui_image_size
    action_dim = max(args.toy_action_dim, ACTION_DIM, ui_action_dim)

    cfg = VAGIConfig(
        vocab_size=256,
        hidden_size=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=args.obs_dim,
        obs_tokens=2,
        action_dim=action_dim,
        memory_slots=4,
        dropout=0.0,
        use_world_pred=(args.mode == "think"),
        use_vision=True,
        vision_channels=args.ui_channels,
    )
    model = VAGICore(cfg)
    model.eval()

    records: List[EpisodeRecord] = []
    records.extend(
        run_toy(
            model,
            seeds,
            args.episodes,
            obs_dim=args.obs_dim,
            action_dim=args.toy_action_dim,
            max_steps=args.max_steps,
            mode=args.mode,
            deterministic=args.deterministic,
        )
    )
    records.extend(
        run_ui(
            model,
            seeds,
            args.episodes,
            image_size=args.ui_image_size,
            action_dim=ui_action_dim,
            max_steps=args.max_steps,
            channels=args.ui_channels,
            mode=args.mode,
            deterministic=args.deterministic,
        )
    )
    records.extend(
        run_code_env(
            model,
            seeds,
            Path(args.tasks_dir),
            obs_dim=args.obs_dim,
            max_steps=args.max_steps,
            mode=args.mode,
            deterministic=args.deterministic,
        )
    )

    by_env: Dict[str, List[EpisodeRecord]] = {}
    for record in records:
        by_env.setdefault(record.env, []).append(record)

    by_task: Dict[str, List[EpisodeRecord]] = {}
    for record in records:
        key = f"{record.env}/{record.task}"
        by_task.setdefault(key, []).append(record)

    report = {
        "config": {
            "episodes": args.episodes,
            "seeds": seeds,
            "max_steps": args.max_steps,
            "obs_dim": args.obs_dim,
            "action_dim": action_dim,
            "mode": args.mode,
        },
        "summary": _aggregate(records),
        "by_env": {env: _aggregate(items) for env, items in by_env.items()},
        "by_task": {task: _aggregate(items) for task, items in by_task.items()},
        "records": [asdict(record) for record in records],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(asdict(r) for r in records)
    print(f"wrote {out_path} and {csv_path}")


if __name__ == "__main__":
    main()
