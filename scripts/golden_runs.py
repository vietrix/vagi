"""Generate deterministic golden benchmark metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from envs.toy_env import ToyEnv
from vagi_core import VAGIConfig, VAGICore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate golden benchmark runs.")
    parser.add_argument("--out-dir", type=str, default="tests/fixtures/golden")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=12)
    return parser.parse_args()


def _build_model(seed: int, obs_dim: int, action_dim: int) -> VAGICore:
    torch.manual_seed(seed)
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
        world_model_horizon=2,
        use_uncertainty=True,
        use_reflection=True,
        use_budget_head=True,
        budget_max_horizon=3,
        budget_max_candidates=4,
    )
    return VAGICore(cfg)


def _run_episode(model: VAGICore, env: ToyEnv, mode: str) -> Dict[str, float]:
    obs = env.reset()
    state = model.init_state(1)
    total_reward = 0.0
    steps = 0
    think_count = 0
    for _ in range(env.max_steps):
        input_ids = torch.zeros((1, 1), dtype=torch.long)
        if mode == "act":
            out = model.act(input_ids=input_ids, obs=obs.unsqueeze(0), state=state)
            action = int(out["action"].item())
            outputs = out["outputs"]
        else:
            plan = model.think_then_act(input_ids=input_ids, obs=obs.unsqueeze(0), state=state)
            action = int(plan["action"].item())
            think_count += 1 if plan["mode"] == "think" else 0
            outputs = model.step(
                input_ids=torch.tensor([[action]], dtype=torch.long),
                obs=obs.unsqueeze(0),
                state=state,
            )

        obs, reward, done, _ = env.step(action)
        total_reward += float(reward)
        state = outputs["state"]
        steps += 1
        if done:
            break
    return {"reward": total_reward, "steps": steps, "think": think_count}


def run_golden(seed: int, episodes: int, steps: int) -> Dict[str, float | int]:
    obs_dim = 8
    action_dim = 4
    model = _build_model(seed, obs_dim, action_dim)
    model.eval()

    act_rewards: List[float] = []
    think_rewards: List[float] = []
    act_steps: List[int] = []
    think_steps: List[int] = []
    think_modes: List[int] = []

    for idx in range(episodes):
        env_act = ToyEnv(obs_dim=obs_dim, action_dim=action_dim, max_steps=steps, seed=seed + idx)
        env_think = ToyEnv(obs_dim=obs_dim, action_dim=action_dim, max_steps=steps, seed=seed + idx)
        act = _run_episode(model, env_act, "act")
        think = _run_episode(model, env_think, "think")
        act_rewards.append(act["reward"])
        think_rewards.append(think["reward"])
        act_steps.append(act["steps"])
        think_steps.append(think["steps"])
        think_modes.append(think["think"])

    def _mean(values: List[float]) -> float:
        return sum(values) / max(len(values), 1)

    payload = {
        "seed": seed,
        "episodes": episodes,
        "act_mean_reward": _mean(act_rewards),
        "think_mean_reward": _mean(think_rewards),
        "act_mean_steps": _mean([float(v) for v in act_steps]),
        "think_mean_steps": _mean([float(v) for v in think_steps]),
        "act_success_rate": sum(1 for v in act_rewards if v > 0.0) / max(len(act_rewards), 1),
        "think_success_rate": sum(1 for v in think_rewards if v > 0.0) / max(len(think_rewards), 1),
        "think_rate": _mean([float(v) for v in think_modes]) / max(steps, 1),
    }
    return payload


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    for seed in seeds:
        payload = run_golden(seed, args.episodes, args.steps)
        (out_dir / f"seed_{seed}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
