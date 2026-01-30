"""Collect rollouts from toy, code, and UI environments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from envs.code_env.actions import ACTION_DIM, ACTION_TYPES
from envs.code_env.code_env import CodeEnv, PATCH_SEPARATOR
from envs.toy_env import ToyEnv
from envs.ui_env import UIEnv
from scripts.baseline_random import action_from_type as action_from_type_random
from scripts.bench_utils import collect_tasks
from scripts.utils import set_deterministic
from utils.data.schema import SCHEMA_VERSION
from vagi_core import RecurrentState, VAGIConfig, VAGICore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect multi-env rollouts.")
    parser.add_argument("--out", type=str, default="logs/multi_env_rollouts.jsonl")
    parser.add_argument("--tasks-dir", type=str, default="envs/code_env/fixtures/benchmarks")
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--episodes-per-env", type=int, default=5)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-run-tests", type=int, default=2)
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--toy-action-dim", type=int, default=4)
    parser.add_argument("--ui-image-size", type=int, default=4)
    parser.add_argument("--ui-channels", type=int, default=1)
    parser.add_argument("--policy", type=str, default="model", choices=["model", "random"])
    parser.add_argument("--mode", type=str, default="act", choices=["act", "think"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _default_patch() -> str:
    old = "def add(a, b):\n    return a - b"
    new = "def add(a, b):\n    return a + b"
    return old + PATCH_SEPARATOR + new


def _load_patch(task_dir: Path) -> str:
    patch_path = task_dir / "solution.patch"
    if patch_path.exists():
        return patch_path.read_text(encoding="utf-8")
    return _default_patch()


def _encode_obs(model: VAGICore, obs: torch.Tensor, *, use_image: bool) -> List[float]:
    if use_image:
        if model.vision is None:
            raise ValueError("Vision encoder is required for image observations")
        with torch.no_grad():
            encoded = model.vision(obs.unsqueeze(0)).squeeze(0)
        return [float(v) for v in encoded.detach().cpu().tolist()]
    return [float(v) for v in obs.detach().cpu().tolist()]


def _select_action(
    model: VAGICore,
    obs: torch.Tensor,
    state: RecurrentState,
    *,
    policy: str,
    mode: str,
    use_image: bool,
) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
    input_ids = torch.zeros((1, 1), dtype=torch.long)
    obs_batch = None if use_image else obs.unsqueeze(0)
    image_batch = obs.unsqueeze(0) if use_image else None

    if policy == "random":
        action = int(torch.randint(0, model.cfg.action_dim, (1,)).item())
        out = model.step(input_ids=input_ids, obs=obs_batch, image=image_batch, state=state)
        return action, out["state"], out["value"]

    if mode == "think":
        plan = model.think_then_act(
            input_ids=input_ids,
            obs=obs_batch,
            image=image_batch,
            state=state,
        )
        action = int(plan["action"].view(-1)[0].item())
        outputs = plan.get("outputs")
        if outputs is None:
            outputs = model.step(input_ids=input_ids, obs=obs_batch, image=image_batch, state=state)
        return action, outputs["state"], outputs["value"]

    out = model.act(input_ids=input_ids, obs=obs_batch, image=image_batch, state=state)
    action = int(out["action"].view(-1)[0].item())
    return action, out["outputs"]["state"], out["outputs"]["value"]


def _aggregate(records: Iterable[Dict[str, object]]) -> Dict[str, float]:
    records = list(records)
    total = len(records)
    if total == 0:
        return {"pass_rate": 0.0, "mean_reward": 0.0, "mean_steps": 0.0}
    pass_rate = sum(1 for r in records if r.get("success")) / total
    mean_reward = sum(float(r.get("total_reward", 0.0)) for r in records) / total
    mean_steps = sum(int(r.get("steps", 0)) for r in records) / total
    return {"pass_rate": pass_rate, "mean_reward": mean_reward, "mean_steps": mean_steps}


def collect_rollouts(
    *,
    out_path: Optional[Path],
    tasks_dir: Path,
    level: Optional[int],
    episodes_per_env: int,
    episodes_per_task: int,
    max_steps: int,
    max_run_tests: int,
    obs_dim: int,
    toy_action_dim: int,
    ui_image_size: int,
    ui_channels: int,
    policy: str,
    mode: str,
    seed: int,
    deterministic: bool,
    model: Optional[VAGICore] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, float]]]:
    if episodes_per_env <= 0:
        raise ValueError("episodes_per_env must be > 0")
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be > 0")

    set_deterministic(seed, deterministic)
    if model is None:
        action_dim = max(toy_action_dim, ACTION_DIM, ui_image_size * ui_image_size)
        cfg = VAGIConfig(
            vocab_size=max(32, action_dim + 1),
            hidden_size=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=4,
            mlp_ratio=2.0,
            max_seq_len=8,
            obs_dim=obs_dim,
            obs_tokens=2,
            action_dim=action_dim,
            memory_slots=4,
            dropout=0.0,
            use_world_pred=(mode == "think"),
            use_vision=True,
            vision_channels=ui_channels,
        )
        model = VAGICore(cfg)
    model.eval()

    records: List[Dict[str, object]] = []
    summaries: Dict[str, List[Dict[str, object]]] = {"toy": [], "ui": [], "code": []}

    # Toy env rollouts
    for idx in range(episodes_per_env):
        env = ToyEnv(obs_dim=obs_dim, action_dim=toy_action_dim, max_steps=max_steps, seed=seed + idx)
        obs = env.reset()
        state = model.init_state(batch_size=1)
        episode_id = f"toy-{seed}-{idx}"
        total_reward = 0.0
        start = time.perf_counter()
        steps = 0
        done = False
        while not done:
            action, state, value = _select_action(
                model, obs, state, policy=policy, mode=mode, use_image=False
            )
            obs_next, reward, done, _info = env.step(action % env.action_dim)
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "episode_id": episode_id,
                    "timestep": steps,
                    "obs": _encode_obs(model, obs, use_image=False),
                    "action": int(action),
                    "reward": float(reward),
                    "done": bool(done),
                    "obs_next": _encode_obs(model, obs_next, use_image=False),
                    "value": float(value.item()) if value is not None else None,
                    "task": "toy",
                    "success": bool(reward > 0.0),
                    "info": {"env": "toy"},
                }
            )
            total_reward += float(reward)
            obs = obs_next
            steps += 1
        latency = time.perf_counter() - start
        summaries["toy"].append(
            {
                "success": total_reward > 0.0,
                "steps": steps,
                "total_reward": total_reward,
                "latency_s": latency,
            }
        )

    # UI env rollouts (image -> obs via vision encoder)
    for idx in range(episodes_per_env):
        env = UIEnv(
            image_size=ui_image_size,
            action_dim=ui_image_size * ui_image_size,
            max_steps=max_steps,
            seed=seed + idx,
            channels=ui_channels,
        )
        obs = env.reset()
        state = model.init_state(batch_size=1)
        episode_id = f"ui-{seed}-{idx}"
        total_reward = 0.0
        start = time.perf_counter()
        steps = 0
        done = False
        while not done:
            action, state, value = _select_action(
                model, obs, state, policy=policy, mode=mode, use_image=True
            )
            obs_next, reward, done, _info = env.step(action % env.action_dim)
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "episode_id": episode_id,
                    "timestep": steps,
                    "obs": _encode_obs(model, obs, use_image=True),
                    "action": int(action),
                    "reward": float(reward),
                    "done": bool(done),
                    "obs_next": _encode_obs(model, obs_next, use_image=True),
                    "value": float(value.item()) if value is not None else None,
                    "task": "ui",
                    "success": bool(reward > 0.0),
                    "info": {"env": "ui"},
                }
            )
            total_reward += float(reward)
            obs = obs_next
            steps += 1
        latency = time.perf_counter() - start
        summaries["ui"].append(
            {
                "success": total_reward > 0.0,
                "steps": steps,
                "total_reward": total_reward,
                "latency_s": latency,
            }
        )

    # Code env rollouts (per task)
    task_dirs = collect_tasks(tasks_dir, level=level)
    for task_dir in task_dirs:
        patch = _load_patch(task_dir)
        for idx in range(episodes_per_task):
            env = CodeEnv(
                obs_dim=obs_dim,
                max_steps=max_steps,
                max_run_tests=max_run_tests,
                seed=seed + idx,
                repo_path=task_dir,
            )
            obs = env.reset()
            state = model.init_state(batch_size=1)
            episode_id = f"code-{task_dir.name}-{seed}-{idx}"
            total_reward = 0.0
            start = time.perf_counter()
            steps = 0
            done = False
            last_info: Dict[str, object] = {}
            while not done:
                action_id, state, value = _select_action(
                    model, obs, state, policy=policy, mode=mode, use_image=False
                )
                action_type = ACTION_TYPES[action_id % len(ACTION_TYPES)]
                action_text = action_from_type_random(action_type, patch)
                obs_next, reward, done, info = env.step(action_text)
                last_info = info
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "episode_id": episode_id,
                        "timestep": steps,
                        "obs": _encode_obs(model, obs, use_image=False),
                        "action": int(action_id),
                        "reward": float(reward),
                        "done": bool(done),
                        "obs_next": _encode_obs(model, obs_next, use_image=False),
                        "value": float(value.item()) if value is not None else None,
                        "task": f"code/{task_dir.name}",
                        "success": bool(info.get("fail_count", 1) == 0),
                        "info": {
                            "env": "code",
                            "fail_count": int(info.get("fail_count", 0)),
                            "action": action_text,
                        },
                    }
                )
                total_reward += float(reward)
                obs = obs_next
                steps += 1
            latency = time.perf_counter() - start
            summaries["code"].append(
                {
                    "success": bool(last_info.get("fail_count", 1) == 0),
                    "steps": steps,
                    "total_reward": total_reward,
                    "latency_s": latency,
                }
            )

    metrics = {env: _aggregate(items) for env, items in summaries.items()}

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    return records, metrics


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    collect_rollouts(
        out_path=out_path,
        tasks_dir=Path(args.tasks_dir),
        level=args.level,
        episodes_per_env=args.episodes_per_env,
        episodes_per_task=args.episodes_per_task,
        max_steps=args.max_steps,
        max_run_tests=args.max_run_tests,
        obs_dim=args.obs_dim,
        toy_action_dim=args.toy_action_dim,
        ui_image_size=args.ui_image_size,
        ui_channels=args.ui_channels,
        policy=args.policy,
        mode=args.mode,
        seed=args.seed,
        deterministic=args.deterministic,
    )
    print(f"Wrote rollouts to {out_path}")


if __name__ == "__main__":
    main()
