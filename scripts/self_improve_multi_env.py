"""Long-run self-improvement loop across toy, code, and UI environments."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch

from envs.code_env.actions import ACTION_DIM
from io.checkpoint import load_checkpoint, save_checkpoint
from scripts.collect_multi_env_rollouts import collect_rollouts
from scripts.utils import get_lr, set_deterministic
from utils.data.pack import pack_batches
from utils.data.schema import RolloutRecord, validate_record
from vagi_core import VAGIConfig, VAGICore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-improve vAGI across multiple environments.")
    parser.add_argument("--out-dir", type=str, default="runs/self_improve_multi_env")
    parser.add_argument("--tasks-dir", type=str, default="envs/code_env/fixtures/benchmarks")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--episodes-per-env", type=int, default=5)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-run-tests", type=int, default=2)
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--toy-action-base", type=int, default=4)
    parser.add_argument("--toy-action-step", type=int, default=2)
    parser.add_argument("--ui-size-base", type=int, default=3)
    parser.add_argument("--ui-size-step", type=int, default=1)
    parser.add_argument("--ui-channels", type=int, default=1)
    parser.add_argument("--start-level", type=int, default=1)
    parser.add_argument("--max-level", type=int, default=3)
    parser.add_argument("--pass-threshold", type=float, default=0.6)
    parser.add_argument("--policy", type=str, default="model", choices=["model", "random"])
    parser.add_argument("--mode", type=str, default="act", choices=["act", "think"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--world-weight", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def _level_config(level: int, args: argparse.Namespace) -> Dict[str, int]:
    return {
        "toy_action_dim": args.toy_action_base + (level - 1) * args.toy_action_step,
        "ui_image_size": args.ui_size_base + (level - 1) * args.ui_size_step,
        "code_level": level,
    }


def _combined_pass(metrics: Dict[str, Dict[str, float]]) -> float:
    total = 0.0
    weight = 0.0
    for env_metrics in metrics.values():
        steps = max(env_metrics.get("mean_steps", 1.0), 1.0)
        weight += steps
        total += env_metrics.get("pass_rate", 0.0) * steps
    if weight == 0.0:
        return 0.0
    return total / weight


def _to_records(raw: List[Dict[str, object]]) -> List[RolloutRecord]:
    records: List[RolloutRecord] = []
    for item in raw:
        records.append(validate_record(item))
    return records


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed, args.deterministic)

    max_toy_action = args.toy_action_base + (args.max_level - 1) * args.toy_action_step
    max_ui_size = args.ui_size_base + (args.max_level - 1) * args.ui_size_step
    action_dim = max(ACTION_DIM, max_toy_action, max_ui_size * max_ui_size)

    cfg = VAGIConfig(
        vocab_size=max(64, action_dim + 1),
        hidden_size=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=args.obs_dim,
        obs_tokens=2,
        action_dim=action_dim,
        memory_slots=4,
        dropout=0.0,
        use_world_pred=args.world_weight > 0.0,
        world_model_horizon=args.horizon,
        use_vision=True,
        vision_channels=args.ui_channels,
    )
    model = VAGICore(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.resume:
        load_checkpoint(model, optimizer=optimizer, ckpt_path=args.resume)

    out_dir = Path(args.out_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, object]] = []
    level = args.start_level

    for iteration in range(args.iterations):
        level_cfg = _level_config(level, args)
        model.eval()
        rollouts_path = run_dir / f"iter_{iteration:03d}.jsonl"
        raw_records, metrics = collect_rollouts(
            out_path=rollouts_path,
            tasks_dir=Path(args.tasks_dir),
            level=level_cfg["code_level"],
            episodes_per_env=args.episodes_per_env,
            episodes_per_task=args.episodes_per_task,
            max_steps=args.max_steps,
            max_run_tests=args.max_run_tests,
            obs_dim=args.obs_dim,
            toy_action_dim=level_cfg["toy_action_dim"],
            ui_image_size=level_cfg["ui_image_size"],
            ui_channels=args.ui_channels,
            policy=args.policy,
            mode=args.mode,
            seed=args.seed + iteration,
            deterministic=args.deterministic,
            model=model,
        )
        records = _to_records(raw_records)
        combined_pass = _combined_pass(metrics)

        model.train()
        step = 0
        start = time.perf_counter()
        for epoch in range(args.epochs):
            for batch in pack_batches(records, batch_size=args.batch_size, horizon=args.horizon, gamma=args.gamma):
                obs = batch["obs"]
                input_ids = torch.zeros((obs.shape[0], 1), dtype=torch.long)
                targets: Dict[str, torch.Tensor] = {
                    "actions": batch["actions"],
                }
                if args.value_weight > 0.0:
                    targets["values"] = batch["returns"]
                if args.world_weight > 0.0:
                    targets["obs_future"] = batch["obs_future"]
                targets["loss_weights"] = {
                    "policy": args.policy_weight,
                    "value": args.value_weight,
                    "world": args.world_weight,
                }
                out = model.forward(input_ids=input_ids, obs=obs, targets=targets, return_loss=True)
                loss = out["loss"]
                if loss is None:
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                if args.log_every > 0 and step % args.log_every == 0:
                    lr = get_lr(optimizer)
                    print(f"iter={iteration} epoch={epoch} step={step} loss={loss.item():.4f} lr={lr:.6f}")

        elapsed = time.perf_counter() - start
        if args.save_every > 0 and (iteration + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, iteration + 1, run_dir)

        history.append(
            {
                "iteration": iteration,
                "level": level,
                "metrics": metrics,
                "combined_pass_rate": combined_pass,
                "train_seconds": elapsed,
            }
        )
        if combined_pass >= args.pass_threshold and level < args.max_level:
            level += 1

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    csv_path = run_dir / "history.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "level", "env", "pass_rate", "mean_reward", "mean_steps", "combined_pass_rate"])
        for entry in history:
            metrics = entry["metrics"]
            for env, env_metrics in metrics.items():
                writer.writerow(
                    [
                        entry["iteration"],
                        entry["level"],
                        env,
                        f"{env_metrics['pass_rate']:.6f}",
                        f"{env_metrics['mean_reward']:.6f}",
                        f"{env_metrics['mean_steps']:.6f}",
                        f"{entry['combined_pass_rate']:.6f}",
                    ]
                )
    print(f"Saved self-improve run to {run_dir}")


if __name__ == "__main__":
    main()
