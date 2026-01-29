"""Collect deterministic rollouts from the toy environment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.distributions import Categorical

from vagi_core import VAGIConfig, VAGICore

from scripts.toy_env import ToyEnv
from scripts.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect rollouts from the toy env.")
    parser.add_argument("--out", type=str, default="logs/rollouts.jsonl")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--obs-tokens", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--memory-slots", type=int, default=4)
    parser.add_argument("--target", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--use-special-tokens", action="store_true", default=True)
    parser.add_argument("--no-special-tokens", action="store_false", dest="use_special_tokens")
    return parser.parse_args()


def _to_list(x: torch.Tensor) -> List[float]:
    return [float(v) for v in x.detach().cpu().tolist()]


def collect_rollouts(
    *,
    out_path: str | Path,
    episodes: int,
    episode_length: int,
    gamma: float,
    seed: int,
    vocab_size: int,
    hidden_size: int,
    layers: int,
    heads: int,
    obs_dim: int,
    obs_tokens: int,
    action_dim: int,
    memory_slots: int,
    target: int,
    max_seq_len: int,
    use_special_tokens: bool,
) -> Dict[str, List[Dict[str, object]]]:
    if episodes <= 0:
        raise ValueError("episodes must be > 0")
    if episode_length <= 0:
        raise ValueError("episode_length must be > 0")

    set_seed(seed)
    device = torch.device("cpu")

    tokens_per_step = 1 + obs_tokens + (3 if use_special_tokens else 0)
    if max_seq_len <= 0:
        max_seq_len = max(8, tokens_per_step * episode_length)

    cfg = VAGIConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_layers=layers,
        n_heads=heads,
        n_kv_heads=heads,
        mlp_ratio=2.0,
        max_seq_len=max_seq_len,
        obs_dim=obs_dim,
        obs_tokens=obs_tokens,
        action_dim=action_dim,
        memory_slots=memory_slots,
        dropout=0.0,
        use_world_pred=False,
        use_special_tokens=use_special_tokens,
    )

    model = VAGICore(cfg).to(device)
    model.eval()

    env = ToyEnv(obs_dim=obs_dim, action_dim=action_dim, max_steps=episode_length, target=target)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".meta.json")

    records: List[Dict[str, object]] = []

    for ep_idx in range(episodes):
        obs = env.reset()
        state = model.init_state(batch_size=1, device=device)
        token_id = 0

        for step_idx in range(episode_length):
            input_ids = torch.tensor([[token_id % cfg.vocab_size]], dtype=torch.long, device=device)
            obs_tensor = obs.unsqueeze(0).to(device)
            out = model.step(input_ids=input_ids, obs=obs_tensor, state=state)

            logits = out["action_logits"]
            dist = Categorical(logits=logits)
            action = int(dist.sample().item())
            log_prob = float(dist.log_prob(torch.tensor(action, device=device)).item())
            value = float(out["value"].squeeze(-1).item())

            step_result = env.step(action)
            next_obs = step_result.obs
            done = bool(step_result.done)

            record = {
                "episode": int(ep_idx),
                "step": int(step_idx),
                "obs": _to_list(obs),
                "action": int(action),
                "reward": float(step_result.reward),
                "done": done,
                "next_obs": _to_list(next_obs),
                "value": value,
                "log_prob": log_prob,
                "input_id": int(token_id),
                "info": step_result.info,
            }
            records.append(record)

            obs = next_obs
            token_id = action
            state = out["state"]

            if done:
                break

    meta = {
        "format_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "episodes": episodes,
        "episode_length": episode_length,
        "gamma": gamma,
        "env": {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "max_steps": episode_length,
            "target": target,
        },
        "model": {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "layers": layers,
            "heads": heads,
            "obs_dim": obs_dim,
            "obs_tokens": obs_tokens,
            "action_dim": action_dim,
            "memory_slots": memory_slots,
            "max_seq_len": max_seq_len,
            "use_special_tokens": use_special_tokens,
        },
        "special_tokens": ["<OBS>", "<ACT>", "<VAL>"] if use_special_tokens else [],
    }

    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {"records": records}


def main() -> None:
    args = parse_args()
    collect_rollouts(
        out_path=args.out,
        episodes=args.episodes,
        episode_length=args.episode_length,
        gamma=args.gamma,
        seed=args.seed,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads,
        obs_dim=args.obs_dim,
        obs_tokens=args.obs_tokens,
        action_dim=args.action_dim,
        memory_slots=args.memory_slots,
        target=args.target,
        max_seq_len=args.max_seq_len,
        use_special_tokens=args.use_special_tokens,
    )
    print(f"Saved rollouts to {args.out}")


if __name__ == "__main__":
    main()
