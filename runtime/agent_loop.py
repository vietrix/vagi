"""Minimal agent loop for vAGI."""

from __future__ import annotations

import argparse
from typing import Optional

import torch

from envs.toy_env import ToyEnv
from runtime.logging import JsonlWriter
from vagi_core import VAGIConfig, VAGICore


def run_episode(
    model: VAGICore,
    env: ToyEnv,
    steps: int,
    log_path: Optional[str] = None,
) -> int:
    model.eval()
    obs = env.reset()
    state = model.init_state(batch_size=1)
    token_id = 0
    writer = JsonlWriter(log_path) if log_path else None

    try:
        for t in range(steps):
            input_ids = torch.tensor([[token_id]], dtype=torch.long)
            out = model.step(input_ids=input_ids, obs=obs.unsqueeze(0), state=state)
            action = int(torch.argmax(out["action_logits"], dim=-1).item())
            value = float(out["value"].item())
            next_obs, reward, done, info = env.step(action)

            if writer is not None:
                writer.write(
                    {
                        "timestep": t,
                        "obs": obs.tolist(),
                        "action": action,
                        "reward": float(reward),
                        "value": value,
                    }
                )

            state = out["state"]
            obs = next_obs
            token_id = action
            if done:
                return t + 1
    finally:
        if writer is not None:
            writer.close()

    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal vAGI agent loop.")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--log", type=str, default="runs/agent/transitions.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-dim", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    env = ToyEnv(obs_dim=args.obs_dim, action_dim=args.action_dim, max_steps=args.steps, seed=args.seed)
    cfg = VAGIConfig(
        vocab_size=32,
        hidden_size=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=args.obs_dim,
        obs_tokens=1,
        action_dim=args.action_dim,
        memory_slots=2,
        dropout=0.0,
        use_world_pred=False,
    )
    model = VAGICore(cfg)

    steps = run_episode(model, env, steps=args.steps, log_path=args.log)
    print(f"Completed {steps} steps. Logs at {args.log}")


if __name__ == "__main__":
    main()
