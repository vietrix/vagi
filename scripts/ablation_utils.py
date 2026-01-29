"""Shared helpers for ablation runs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import torch

from envs.code_env.actions import ACTION_DIM, ACTION_TYPES
from envs.code_env.code_env import CodeEnv, PATCH_SEPARATOR
from scripts.baseline_random import action_from_type as action_from_type_random
from scripts.utils import set_deterministic
from vagi_core import VAGIConfig, VAGICore
from vagi_core.memory import KVCache, RecurrentState


def default_patch() -> str:
    old = "def add(a, b):\n    return a - b"
    new = "def add(a, b):\n    return a + b"
    return old + PATCH_SEPARATOR + new


def run_vagi_records(
    *,
    tasks: List[Path],
    seeds: List[int],
    episodes_per_task: int,
    obs_dim: int,
    max_steps: int,
    max_run_tests: int,
    deterministic: bool,
    config_overrides: Dict[str, object] | None = None,
    use_kv_cache: bool = True,
) -> List[Dict[str, object]]:
    config_overrides = config_overrides or {}
    cfg = VAGIConfig(
        vocab_size=256,
        hidden_size=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=obs_dim,
        obs_tokens=2,
        action_dim=ACTION_DIM,
        memory_slots=4,
        dropout=0.0,
        use_world_pred=False,
    )
    for key, value in config_overrides.items():
        setattr(cfg, key, value)
    cfg.validate()
    model = VAGICore(cfg)
    model.eval()

    patch = default_patch()
    records: List[Dict[str, object]] = []
    for seed in seeds:
        set_deterministic(seed, deterministic)
        for task_dir in tasks:
            for episode_idx in range(episodes_per_task):
                episode_seed = seed + episode_idx
                env = CodeEnv(
                    obs_dim=obs_dim,
                    max_steps=max_steps,
                    max_run_tests=max_run_tests,
                    seed=episode_seed,
                    repo_path=task_dir,
                )
                obs = env.reset()
                state = model.init_state(batch_size=1)
                total_reward = 0.0
                info: Dict[str, object] = {}
                steps = 0
                start = time.perf_counter()
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
                    if not use_kv_cache:
                        state = RecurrentState(
                            mem=state.mem,
                            kv=KVCache.empty(cfg.n_layers),
                            timestep=state.timestep,
                        )
                    if done:
                        break
                latency = time.perf_counter() - start
                success = int(info.get("fail_count", 1)) == 0
                records.append(
                    {
                        "task": task_dir.name,
                        "seed": episode_seed,
                        "success": success,
                        "steps": steps,
                        "total_reward": total_reward,
                        "latency_s": latency,
                    }
                )
    return records


def aggregate(records: List[Dict[str, object]]) -> Dict[str, float]:
    total = len(records)
    if total == 0:
        return {"pass_rate": 0.0, "mean_reward": 0.0, "mean_steps": 0.0, "mean_latency_s": 0.0}
    pass_rate = sum(1 for r in records if r.get("success")) / total
    mean_reward = sum(float(r["total_reward"]) for r in records) / total
    mean_steps = sum(float(r["steps"]) for r in records) / total
    mean_latency = sum(float(r["latency_s"]) for r in records) / total
    return {
        "pass_rate": pass_rate,
        "mean_reward": mean_reward,
        "mean_steps": mean_steps,
        "mean_latency_s": mean_latency,
    }


def aggregate_by_task(records: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        task = str(record.get("task", "unknown"))
        buckets.setdefault(task, []).append(record)
    summary: Dict[str, Dict[str, float]] = {}
    for task, recs in buckets.items():
        metrics = aggregate(recs)
        metrics["episodes"] = float(len(recs))
        summary[task] = metrics
    return summary
