"""Run a minimal agent loop using vAGI and the toy environment."""

from __future__ import annotations

import argparse
from typing import Dict, List

import torch

from vagi_core import VAGIConfig, VAGICore

from scripts.toy_env import ToyEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a toy agent loop with vAGI.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None, help="Save transitions to a .pt file")
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--obs-tokens", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--memory-slots", type=int, default=4)
    parser.add_argument("--target", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = VAGIConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.heads,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=args.obs_dim,
        obs_tokens=args.obs_tokens,
        action_dim=args.action_dim,
        memory_slots=args.memory_slots,
        dropout=0.0,
        use_world_pred=False,
    )
    model = VAGICore(cfg).to(device)
    model.eval()

    env = ToyEnv(obs_dim=args.obs_dim, action_dim=args.action_dim, max_steps=args.max_steps, target=args.target)
    obs = env.reset()
    state = model.init_state(batch_size=1, device=device)

    transitions: List[Dict[str, torch.Tensor]] = []
    token_id = 0

    for step_idx in range(args.steps):
        input_ids = torch.tensor([[token_id % cfg.vocab_size]], dtype=torch.long, device=device)
        obs_tensor = obs.unsqueeze(0).to(device)
        out = model.step(input_ids=input_ids, obs=obs_tensor, state=state)

        action_logits = out["action_logits"]
        action = int(torch.argmax(action_logits, dim=-1).item())

        step_result = env.step(action)
        next_obs = step_result.obs
        done = step_result.done

        transitions.append(
            {
                "input_ids": input_ids.squeeze(0).cpu(),
                "obs": obs.cpu(),
                "actions": torch.tensor(action, dtype=torch.long),
                "rewards": torch.tensor(step_result.reward, dtype=torch.float32),
                "dones": torch.tensor(done, dtype=torch.bool),
                "next_obs": next_obs.cpu(),
            }
        )

        state = out["state"]
        obs = next_obs
        token_id = action

        print(
            f"step={step_idx + 1} action={action} reward={step_result.reward:.2f} "
            f"done={done} pos={step_result.info['position']}"
        )
        if done:
            break

    if args.save:
        stacked = {
            "input_ids": torch.stack([t["input_ids"] for t in transitions], dim=0),
            "obs": torch.stack([t["obs"] for t in transitions], dim=0),
            "actions": torch.stack([t["actions"] for t in transitions], dim=0),
            "rewards": torch.stack([t["rewards"] for t in transitions], dim=0),
            "dones": torch.stack([t["dones"] for t in transitions], dim=0),
            "next_obs": torch.stack([t["next_obs"] for t in transitions], dim=0),
        }
        torch.save(stacked, args.save)
        print(f"Saved transitions to {args.save}")


if __name__ == "__main__":
    main()
