"""Run consolidation distillation schedule with forgetting diagnostics."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F

from envs.toy_env import ToyEnv
from io.checkpoint import load_checkpoint, save_checkpoint
from scripts.utils import get_lr, set_deterministic
from utils.data.pack import pack_batches
from utils.data.reader import read_jsonl
from vagi_core import VAGIConfig, VAGICore
from vagi_core.diagnostics import compute_drop, should_rollback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidation schedule with anti-forgetting.")
    parser.add_argument("--new-data", type=str, required=True)
    parser.add_argument("--replay-data", type=str, required=True)
    parser.add_argument("--teacher", type=str, required=True)
    parser.add_argument("--anchor", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="runs/consolidation")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--replay-ratio", type=float, default=0.5)
    parser.add_argument("--anchor-l2", type=float, default=0.0)
    parser.add_argument("--drop-threshold", type=float, default=0.5)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--obs-tokens", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--memory-slots", type=int, default=4)
    parser.add_argument("--use-world", action="store_true")
    parser.add_argument("--use-uncertainty", action="store_true")
    return parser.parse_args()


def _distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temp = max(temperature, 1e-6)
    student_logp = F.log_softmax(student_logits / temp, dim=-1)
    teacher_p = F.softmax(teacher_logits / temp, dim=-1)
    return F.kl_div(student_logp, teacher_p, reduction="batchmean") * (temp ** 2)


def _mix_data(new_path: Path, replay_path: Path, out_path: Path, ratio: float, seed: int) -> int:
    if ratio <= 0.0:
        out_path.write_text(new_path.read_text(encoding="utf-8"), encoding="utf-8")
        return len(out_path.read_text(encoding="utf-8").splitlines())
    if ratio >= 1.0:
        out_path.write_text(replay_path.read_text(encoding="utf-8"), encoding="utf-8")
        return len(out_path.read_text(encoding="utf-8").splitlines())

    new_lines = [ln for ln in new_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    replay_lines = [ln for ln in replay_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not new_lines or not replay_lines:
        out_path.write_text("\n".join(new_lines + replay_lines) + "\n", encoding="utf-8")
        return len(new_lines) + len(replay_lines)

    rng = random.Random(seed)
    max_total = min(len(new_lines) / max(1e-6, (1.0 - ratio)), len(replay_lines) / max(1e-6, ratio))
    total = max(1, int(max_total))
    replay_count = max(1, int(total * ratio))
    new_count = max(1, total - replay_count)
    new_sel = rng.sample(new_lines, min(new_count, len(new_lines)))
    replay_sel = rng.sample(replay_lines, min(replay_count, len(replay_lines)))
    mixed = new_sel + replay_sel
    rng.shuffle(mixed)
    out_path.write_text("\n".join(mixed) + "\n", encoding="utf-8")
    return len(mixed)


def _evaluate_model(
    model: VAGICore,
    seeds: List[int],
    episodes: int,
    steps: int,
    mode: str,
) -> Dict[str, float]:
    rewards = []
    for seed in seeds:
        for idx in range(episodes):
            env = ToyEnv(obs_dim=model.cfg.obs_dim, action_dim=model.cfg.action_dim, max_steps=steps, seed=seed + idx)
            obs = env.reset()
            state = model.init_state(1)
            total = 0.0
            for _ in range(steps):
                input_ids = torch.zeros((1, 1), dtype=torch.long)
                if mode == "think" and model.world is not None:
                    out = model.think_then_act(input_ids=input_ids, obs=obs.unsqueeze(0), state=state)
                    action = int(out["action"].item())
                    step_out = model.step(
                        input_ids=torch.tensor([[action]], dtype=torch.long),
                        obs=obs.unsqueeze(0),
                        state=state,
                    )
                    state = step_out["state"]
                else:
                    out = model.act(input_ids=input_ids, obs=obs.unsqueeze(0), state=state)
                    action = int(out["action"].item())
                    state = out["outputs"]["state"]
                obs, reward, done, _info = env.step(action)
                total += float(reward)
                if done:
                    break
            rewards.append(total)
    mean_reward = sum(rewards) / max(len(rewards), 1)
    return {"mean_reward": mean_reward}


def _evaluate_golden(model: VAGICore, seeds: List[int], episodes: int, steps: int) -> Dict[str, float]:
    act = _evaluate_model(model, seeds, episodes, steps, mode="act")
    think = _evaluate_model(model, seeds, episodes, steps, mode="think")
    return {
        "act_mean_reward": act["mean_reward"],
        "think_mean_reward": think["mean_reward"],
    }


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed, args.deterministic)

    use_world = args.use_world
    use_uncertainty = args.use_uncertainty
    cfg = VAGIConfig(
        vocab_size=max(args.vocab_size, args.action_dim + 1),
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
        use_world_pred=use_world,
        world_model_horizon=args.horizon,
        use_uncertainty=use_uncertainty,
    )
    teacher = VAGICore(cfg)
    load_checkpoint(teacher, optimizer=None, ckpt_path=args.teacher)
    teacher.eval()

    anchor = None
    if args.anchor:
        anchor = VAGICore(cfg)
        load_checkpoint(anchor, optimizer=None, ckpt_path=args.anchor)
        anchor.eval()

    student = VAGICore(cfg)
    student.load_state_dict(teacher.state_dict())
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, float]] = []

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    baseline_metrics = None
    if not args.skip_eval:
        baseline_metrics = _evaluate_golden(teacher, seeds, args.episodes, args.steps)

    for it in range(args.iterations):
        mixed_path = out_dir / f"mixed_iter_{it}.jsonl"
        _mix_data(Path(args.new_data), Path(args.replay_data), mixed_path, args.replay_ratio, args.seed + it)

        step = 0
        for _ in range(args.epochs):
            batch_iter = pack_batches(
                read_jsonl(mixed_path),
                batch_size=args.batch_size,
                horizon=args.horizon,
                gamma=args.gamma,
            )
            for batch in batch_iter:
                obs = batch["obs"]
                actions = batch["actions"]
                input_ids = actions.unsqueeze(1).clamp(max=cfg.vocab_size - 1)
                state = student.init_state(batch_size=obs.shape[0])

                with torch.no_grad():
                    teacher_out = teacher.forward(input_ids=input_ids, obs=obs, state=None, return_loss=False)
                    anchor_out = None
                    if anchor is not None:
                        anchor_out = anchor.forward(input_ids=input_ids, obs=obs, state=None, return_loss=False)

                student_out = student.forward(input_ids=input_ids, obs=obs, state=state, return_loss=False)
                loss = _distill_loss(student_out["action_logits"], teacher_out["action_logits"], args.temperature)
                if teacher_out.get("value") is not None:
                    loss = loss + 0.5 * F.mse_loss(student_out["value"], teacher_out["value"])
                if use_world and teacher_out.get("world_pred") is not None and student_out.get("world_pred") is not None:
                    loss = loss + 0.5 * F.mse_loss(student_out["world_pred"], teacher_out["world_pred"])
                if use_uncertainty and teacher_out.get("world_logvar") is not None and student_out.get("world_logvar") is not None:
                    loss = loss + 0.1 * F.mse_loss(student_out["world_logvar"], teacher_out["world_logvar"])
                if args.anchor_l2 > 0.0 and anchor is not None:
                    l2 = 0.0
                    for p, p_anchor in zip(student.parameters(), anchor.parameters()):
                        l2 = l2 + torch.mean((p - p_anchor) ** 2)
                    loss = loss + args.anchor_l2 * l2

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                step += 1
                if step % 25 == 0:
                    print(f"iter={it} step={step} loss={loss.item():.6f} lr={get_lr(optimizer):.6f}")

        save_checkpoint(student, optimizer, step=step, out_dir=out_dir / f"iter_{it}")

        if args.skip_eval:
            continue
        metrics = _evaluate_golden(student, seeds, args.episodes, args.steps)
        drop = compute_drop(baseline_metrics, metrics, key="act_mean_reward")
        record = {
            "iteration": float(it),
            "act_mean_reward": float(metrics["act_mean_reward"]),
            "think_mean_reward": float(metrics["think_mean_reward"]),
            "drop": float(drop),
            "lr": get_lr(optimizer),
            "replay_ratio": float(args.replay_ratio),
        }
        history.append(record)

        if should_rollback(drop, args.drop_threshold):
            if args.anchor:
                load_checkpoint(student, optimizer=None, ckpt_path=args.anchor)
            else:
                student.load_state_dict(teacher.state_dict())
            for group in optimizer.param_groups:
                group["lr"] = group["lr"] * 0.5
            args.replay_ratio = min(0.9, args.replay_ratio + 0.1)
            record["rollback"] = 1.0
        else:
            teacher.load_state_dict(student.state_dict())
            record["rollback"] = 0.0

        history_path = out_dir / "forgetting_history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    summary = {
        "iterations": args.iterations,
        "history": history,
        "timestamp": time.time(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
