"""Counterfactual evaluation for policy-only vs planning."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

from envs.toy_env import ToyEnv
from runtime.logging import JsonlWriter
from scripts.utils import set_deterministic
from vagi_core import VAGIConfig, VAGICore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Counterfactual evaluation for budget tuning.")
    parser.add_argument("--out", type=str, default="logs/counterfactual.jsonl")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-dim", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _value_spread(candidate_values: torch.Tensor | None) -> float:
    if candidate_values is None:
        return 0.0
    if candidate_values.ndim == 2:
        vals = candidate_values
    else:
        vals = candidate_values.view(candidate_values.shape[0], -1)
    return float(vals.std(dim=-1).mean().item())


def run_counterfactual(
    *,
    out_path: str | Path,
    episodes: int,
    steps: int,
    seed: int,
    obs_dim: int,
    action_dim: int,
    horizon: int,
    num_candidates: int,
    deterministic: bool,
) -> List[Dict[str, float]]:
    if episodes <= 0:
        raise ValueError("episodes must be > 0")
    if steps <= 0:
        raise ValueError("steps must be > 0")

    set_deterministic(seed, deterministic)
    device = torch.device("cpu")

    cfg = VAGIConfig(
        vocab_size=32,
        hidden_size=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=obs_dim,
        obs_tokens=1,
        action_dim=action_dim,
        memory_slots=2,
        dropout=0.0,
        use_world_pred=True,
        world_model_horizon=1,
        use_uncertainty=True,
    )
    model = VAGICore(cfg).to(device)
    model.eval()

    env = ToyEnv(obs_dim=obs_dim, action_dim=action_dim, max_steps=steps, seed=seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, float]] = []
    with JsonlWriter(out_path) as writer:
        for ep in range(episodes):
            obs = env.reset()
            state = model.init_state(batch_size=1, device=device)
            token_id = 0
            for t in range(steps):
                input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)

                start = time.perf_counter()
                act_out = model.act(input_ids=input_ids, obs=obs.unsqueeze(0), state=state)
                act_latency = time.perf_counter() - start
                act_action = int(act_out["action"].item())

                start = time.perf_counter()
                plan_out = model.plan_step(
                    input_ids=input_ids,
                    obs=obs.unsqueeze(0),
                    state=state,
                    num_candidates=num_candidates,
                    horizon=horizon,
                    trace=True,
                )
                plan_latency = time.perf_counter() - start
                plan_action = int(plan_out["action"].item())

                target = env._target_action(obs)
                act_reward = 1.0 if act_action == target else 0.0
                plan_reward = 1.0 if plan_action == target else 0.0

                record = {
                    "episode": float(ep),
                    "timestep": float(t),
                    "uncertainty": float(plan_out["uncertainty"][0][0].item()),
                    "confidence": float(plan_out["confidence"][0][0].item()),
                    "value_spread": _value_spread(plan_out.get("candidate_values")),
                    "task_difficulty": 0.5,
                    "act_reward": act_reward,
                    "plan_reward": plan_reward,
                    "delta_reward": plan_reward - act_reward,
                    "act_latency": act_latency,
                    "plan_latency": plan_latency,
                    "delta_latency": plan_latency - act_latency,
                }
                writer.write(record)
                records.append(record)

                obs, _reward, done, _info = env.step(plan_action)
                step_out = model.step(
                    input_ids=torch.tensor([[plan_action]], dtype=torch.long, device=device),
                    obs=obs.unsqueeze(0),
                    state=state,
                )
                state = step_out["state"]
                token_id = plan_action
                if done:
                    break

    return records


def main() -> None:
    args = parse_args()
    records = run_counterfactual(
        out_path=args.out,
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        horizon=args.horizon,
        num_candidates=args.num_candidates,
        deterministic=args.deterministic,
    )
    summary = {
        "records": len(records),
        "mean_delta_reward": sum(r["delta_reward"] for r in records) / max(len(records), 1),
        "mean_delta_latency": sum(r["delta_latency"] for r in records) / max(len(records), 1),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
