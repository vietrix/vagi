"""Ablation: compare world head on/off."""

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
    parser = argparse.ArgumentParser(description="Ablate world head for the toy env.")
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
    parser.add_argument("--out-dir", type=str, default="runs")
    parser.add_argument("--use-special-tokens", action="store_true", default=True)
    parser.add_argument("--no-special-tokens", action="store_false", dest="use_special_tokens")
    return parser.parse_args()


def _run_episode(env: ToyEnv, model: VAGICore, episode_length: int) -> Dict[str, float]:
    obs = env.reset()
    state = model.init_state(batch_size=1, device="cpu")
    token_id = 0
    total_reward = 0.0
    total_mse = 0.0
    mse_count = 0
    for _ in range(episode_length):
        input_ids = torch.tensor([[token_id % model.cfg.vocab_size]], dtype=torch.long)
        obs_tensor = obs.unsqueeze(0)
        out = model.step(input_ids=input_ids, obs=obs_tensor, state=state)
        logits = out["action_logits"]
        dist = Categorical(logits=logits)
        action = int(dist.sample().item())

        step_result = env.step(action)
        total_reward += float(step_result.reward)

        if out["world_pred"] is not None:
            pred = out["world_pred"]
            if pred.ndim == 3:
                pred = pred[:, 0, :]
            pred = pred.squeeze(0)
            target = step_result.obs
            total_mse += float(torch.mean((pred - target) ** 2).item())
            mse_count += 1

        obs = step_result.obs
        state = out["state"]
        token_id = action
        if step_result.done:
            break
    avg_mse = total_mse / max(mse_count, 1)
    return {"reward": total_reward, "world_mse": avg_mse}


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
    use_special_tokens: bool,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for use_world in (False, True):
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
            memory_slots=memory_slots,
            dropout=0.0,
            use_world_pred=use_world,
            use_special_tokens=use_special_tokens,
        )
        set_seed(seed)
        model = VAGICore(cfg)
        model.eval()
        env = ToyEnv(obs_dim=obs_dim, action_dim=action_dim, max_steps=episode_length, target=target)
        total_reward = 0.0
        total_mse = 0.0
        for _ in range(episodes):
            metrics = _run_episode(env, model, episode_length)
            total_reward += metrics["reward"]
            total_mse += metrics["world_mse"]
        results.append(
            {
                "use_world_pred": float(1 if use_world else 0),
                "avg_reward": total_reward / max(episodes, 1),
                "avg_world_mse": total_mse / max(episodes, 1),
            }
        )
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
        use_special_tokens=args.use_special_tokens,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ablate_world_{stamp}.json"
    out_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    for entry in results:
        print(
            "use_world_pred={flag} avg_reward={reward:.3f} avg_world_mse={mse:.4f}".format(
                flag=int(entry["use_world_pred"]),
                reward=entry["avg_reward"],
                mse=entry["avg_world_mse"],
            )
        )


if __name__ == "__main__":
    main()
