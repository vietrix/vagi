"""Evaluate greedy policy vs risk-aware planning."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from envs.toy_env import ToyEnv
from vagi_core import VAGIConfig, VAGICore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate planning-only behavior.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-dim", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="results/planning_eval.json")
    return parser.parse_args()


def _build_model(obs_dim: int, action_dim: int, seed: int) -> VAGICore:
    torch.manual_seed(seed)
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
        use_world_pred=True,
        world_model_horizon=max(1, int(3)),
        use_uncertainty=True,
    )
    return VAGICore(cfg)


def _episode_rollout(
    model: VAGICore,
    env: ToyEnv,
    *,
    steps: int,
    mode: str,
    num_candidates: int,
    horizon: int,
    uncertainty_weight: float,
) -> Tuple[float, int, float]:
    obs = env.reset()
    state = model.init_state(batch_size=1)
    total_reward = 0.0
    total_uncertainty = 0.0
    count_uncertainty = 0

    for _ in range(steps):
        input_ids = torch.zeros((1, 1), dtype=torch.long)
        if mode == "plan":
            plan = model.plan_step(
                input_ids=input_ids,
                obs=obs.unsqueeze(0),
                state=state,
                num_candidates=num_candidates,
                horizon=horizon,
                uncertainty_weight=uncertainty_weight,
            )
            action = int(plan["action"].item())
            out = model.step(input_ids=torch.tensor([[action]], dtype=torch.long), obs=obs.unsqueeze(0), state=state)
        else:
            out = model.step(input_ids=input_ids, obs=obs.unsqueeze(0), state=state)
            action = int(torch.argmax(out["action_logits"], dim=-1).item())

        world_logvar = out.get("world_logvar")
        if world_logvar is not None:
            if world_logvar.ndim == 3:
                step_uncertainty = torch.exp(world_logvar[:, 0, :]).mean().item()
            else:
                step_uncertainty = torch.exp(world_logvar).mean().item()
            total_uncertainty += float(step_uncertainty)
            count_uncertainty += 1

        obs, reward, done, _ = env.step(action)
        total_reward += float(reward)
        state = out["state"]
        if done:
            break

    mean_uncertainty = total_uncertainty / max(count_uncertainty, 1)
    return total_reward, env.step_count, mean_uncertainty


def _aggregate(metrics: List[Tuple[float, int, float]]) -> Dict[str, float]:
    total_reward = sum(m[0] for m in metrics)
    total_steps = sum(m[1] for m in metrics)
    total_unc = sum(m[2] for m in metrics)
    episodes = max(len(metrics), 1)
    return {
        "episodes": float(episodes),
        "mean_reward": total_reward / episodes,
        "mean_steps": total_steps / episodes,
        "mean_uncertainty": total_unc / episodes,
        "success_rate": sum(1 for m in metrics if m[0] > 0.0) / episodes,
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_model = _build_model(args.obs_dim, args.action_dim, seed=args.seed)
    model_greedy = _build_model(args.obs_dim, args.action_dim, seed=args.seed)
    model_plan = _build_model(args.obs_dim, args.action_dim, seed=args.seed)
    model_greedy.load_state_dict(base_model.state_dict())
    model_plan.load_state_dict(base_model.state_dict())

    greedy_metrics: List[Tuple[float, int, float]] = []
    plan_metrics: List[Tuple[float, int, float]] = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        env_greedy = ToyEnv(obs_dim=args.obs_dim, action_dim=args.action_dim, max_steps=args.steps, seed=seed)
        env_plan = ToyEnv(obs_dim=args.obs_dim, action_dim=args.action_dim, max_steps=args.steps, seed=seed)
        greedy_metrics.append(
            _episode_rollout(
                model_greedy,
                env_greedy,
                steps=args.steps,
                mode="greedy",
                num_candidates=args.num_candidates,
                horizon=args.horizon,
                uncertainty_weight=args.uncertainty_weight,
            )
        )
        plan_metrics.append(
            _episode_rollout(
                model_plan,
                env_plan,
                steps=args.steps,
                mode="plan",
                num_candidates=args.num_candidates,
                horizon=args.horizon,
                uncertainty_weight=args.uncertainty_weight,
            )
        )

    greedy = _aggregate(greedy_metrics)
    plan = _aggregate(plan_metrics)
    payload = {
        "config": {
            "episodes": args.episodes,
            "steps": args.steps,
            "obs_dim": args.obs_dim,
            "action_dim": args.action_dim,
            "num_candidates": args.num_candidates,
            "horizon": args.horizon,
            "uncertainty_weight": args.uncertainty_weight,
            "seed": args.seed,
        },
        "greedy": greedy,
        "plan": plan,
        "uncertainty_penalty_effectiveness": greedy["mean_uncertainty"] - plan["mean_uncertainty"],
        "timestamp": time.time(),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
