"""Benchmark ablations for fast memory and world head."""

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
    parser = argparse.ArgumentParser(description="Benchmark ablations for vAGI.")
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


def _run_config(
    *,
    cfg: VAGIConfig,
    episodes: int,
    episode_length: int,
    target: int,
    seed: int,
) -> Dict[str, float]:
    device = torch.device("cpu")
    set_seed(seed)
    model = VAGICore(cfg).to(device)
    model.eval()
    env = ToyEnv(obs_dim=cfg.obs_dim, action_dim=cfg.action_dim, max_steps=episode_length, target=target)

    total_reward = 0.0
    total_steps = 0
    total_world_mse = 0.0
    world_count = 0
    start = time.perf_counter()

    for _ in range(episodes):
        obs = env.reset()
        state = model.init_state(batch_size=1, device=device)
        token_id = 0
        for _ in range(episode_length):
            input_ids = torch.tensor([[token_id % cfg.vocab_size]], dtype=torch.long, device=device)
            obs_tensor = obs.unsqueeze(0).to(device)
            out = model.step(input_ids=input_ids, obs=obs_tensor, state=state)
            dist = Categorical(logits=out["action_logits"])
            action = int(dist.sample().item())

            step_result = env.step(action)
            total_reward += float(step_result.reward)
            total_steps += 1

            if out["world_pred"] is not None:
                world_pred = out["world_pred"]
                if world_pred.ndim == 3:
                    world_pred = world_pred[:, 0, :]
                world_pred = world_pred.squeeze(0)
                target_obs = step_result.obs.to(device)
                total_world_mse += float(torch.mean((world_pred - target_obs) ** 2).item())
                world_count += 1

            obs = step_result.obs
            state = out["state"]
            token_id = action
            if step_result.done:
                break

    elapsed = time.perf_counter() - start
    steps_per_sec = total_steps / max(elapsed, 1e-6)
    avg_reward = total_reward / max(episodes, 1)
    avg_world_mse = total_world_mse / max(world_count, 1)
    return {
        "avg_reward": float(avg_reward),
        "steps_per_sec": float(steps_per_sec),
        "avg_world_mse": float(avg_world_mse),
        "steps": float(total_steps),
    }


def _plot_runs(results: List[Dict[str, object]], out_path: Path, key: str) -> None:
    lines = [f"Metric: {key}"]
    max_val = max(float(r.get(key, 0.0)) for r in results) if results else 0.0
    for result in results:
        label = result["label"]
        value = float(result.get(key, 0.0))
        bar_len = 1 if max_val <= 0 else max(1, int((value / max_val) * 40))
        bar = "#" * bar_len
        lines.append(f"{label:>20} | {bar} {value:.3f}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combinations = [
        {"memory_slots": 0, "use_world_pred": False},
        {"memory_slots": 0, "use_world_pred": True},
        {"memory_slots": args.memory_slots, "use_world_pred": False},
        {"memory_slots": args.memory_slots, "use_world_pred": True},
    ]

    results: List[Dict[str, object]] = []
    for combo in combinations:
        tokens_per_step = 1 + args.obs_tokens + (3 if args.use_special_tokens else 0)
        max_seq_len = max(8, tokens_per_step * args.episode_length)
        cfg = VAGIConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            n_layers=args.layers,
            n_heads=args.heads,
            n_kv_heads=args.heads,
            mlp_ratio=2.0,
            max_seq_len=max_seq_len,
            obs_dim=args.obs_dim,
            obs_tokens=args.obs_tokens,
            action_dim=args.action_dim,
            memory_slots=combo["memory_slots"],
            dropout=0.0,
            use_world_pred=combo["use_world_pred"],
            use_special_tokens=args.use_special_tokens,
        )
        label = f"mem={combo['memory_slots']} world={int(combo['use_world_pred'])}"
        metrics = _run_config(
            cfg=cfg,
            episodes=args.episodes,
            episode_length=args.episode_length,
            target=args.target,
            seed=args.seed,
        )
        metrics.update(
            {
                "label": label,
                "memory_slots": int(combo["memory_slots"]),
                "use_world_pred": bool(combo["use_world_pred"]),
            }
        )
        results.append(metrics)
        print(f"{label} reward={metrics['avg_reward']:.3f} steps/s={metrics['steps_per_sec']:.1f}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"benchmark_ablations_{stamp}.json"
    out_path.write_text(json.dumps({"results": results, "seed": args.seed}, indent=2), encoding="utf-8")

    plot_path = out_dir / f"benchmark_ablations_{stamp}_plot.txt"
    _plot_runs(results, plot_path, key="steps_per_sec")


if __name__ == "__main__":
    main()
