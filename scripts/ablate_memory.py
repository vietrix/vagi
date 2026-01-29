"""Ablation: compare memory slots on/off."""

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
    parser = argparse.ArgumentParser(description="Ablate memory slots for the toy env.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=16)
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
    parser.add_argument("--policy", type=str, default="model", choices=["model", "heuristic"])
    parser.add_argument("--out-dir", type=str, default="runs")
    parser.add_argument("--use-special-tokens", action="store_true", default=True)
    parser.add_argument("--no-special-tokens", action="store_false", dest="use_special_tokens")
    return parser.parse_args()


def _heuristic_action(obs: torch.Tensor) -> int:
    pos = int(torch.round(obs[0]).item())
    target = int(torch.round(obs[1]).item())
    if pos < target:
        return 1
    if pos > target:
        return 2
    return 0


def _run_episode(
    env: ToyEnv,
    model: VAGICore,
    policy: str,
    seed: int,
    episode_length: int,
) -> float:
    obs = env.reset()
    state = model.init_state(batch_size=1, device="cpu")
    token_id = 0
    total_reward = 0.0
    for _ in range(episode_length):
        if policy == "heuristic":
            action = _heuristic_action(obs)
        else:
            input_ids = torch.tensor([[token_id % model.cfg.vocab_size]], dtype=torch.long)
            obs_tensor = obs.unsqueeze(0)
            out = model.step(input_ids=input_ids, obs=obs_tensor, state=state)
            logits = out["action_logits"]
            dist = Categorical(logits=logits)
            action = int(dist.sample().item())
            state = out["state"]
        step_result = env.step(action)
        total_reward += float(step_result.reward)
        obs = step_result.obs
        token_id = action
        if step_result.done:
            break
    return total_reward


def run_ablation(
    *,
    episodes: int,
    episode_length: int,
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
    policy: str,
    use_special_tokens: bool,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for slots in (0, memory_slots):
        tokens_per_step = 1 + obs_tokens + (3 if use_special_tokens else 0)
        cfg = VAGIConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=layers,
            n_heads=heads,
            n_kv_heads=heads,
            mlp_ratio=2.0,
            max_seq_len=max(8, tokens_per_step * episode_length),
            obs_dim=obs_dim,
            obs_tokens=obs_tokens,
            action_dim=action_dim,
            memory_slots=slots,
            dropout=0.0,
            use_world_pred=False,
            use_special_tokens=use_special_tokens,
        )
        set_seed(seed)
        model = VAGICore(cfg)
        model.eval()
        env = ToyEnv(obs_dim=obs_dim, action_dim=action_dim, max_steps=episode_length, target=target)
        total = 0.0
        for _ in range(episodes):
            total += _run_episode(env, model, policy, seed, episode_length)
        results.append({"memory_slots": float(slots), "avg_reward": total / max(episodes, 1)})
    return results


def main() -> None:
    args = parse_args()
    results = run_ablation(
        episodes=args.episodes,
        episode_length=args.episode_length,
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
        policy=args.policy,
        use_special_tokens=args.use_special_tokens,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ablate_memory_{stamp}.json"
    out_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    for entry in results:
        print(f"memory_slots={int(entry['memory_slots'])} avg_reward={entry['avg_reward']:.3f}")


if __name__ == "__main__":
    main()
