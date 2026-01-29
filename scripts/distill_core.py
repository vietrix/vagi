"""Distill a teacher vAGI model into a smaller student."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
from torch.nn import functional as F

from io.checkpoint import load_checkpoint, save_checkpoint
from scripts.utils import get_lr, set_deterministic
from utils.data.pack import pack_batches
from utils.data.reader import read_jsonl
from vagi_core import VAGIConfig, VAGICore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill vAGI to a smaller backbone.")
    parser.add_argument("--data", type=str, default="logs/rollouts.jsonl")
    parser.add_argument("--teacher", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="runs/distill")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--world-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--obs-tokens", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--memory-slots", type=int, default=2)
    return parser.parse_args()


def _distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    temp = max(temperature, 1e-6)
    student_logp = F.log_softmax(student_logits / temp, dim=-1)
    teacher_p = F.softmax(teacher_logits / temp, dim=-1)
    return F.kl_div(student_logp, teacher_p, reduction="batchmean") * (temp ** 2)


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed, args.deterministic)

    teacher_cfg = VAGIConfig(
        vocab_size=max(args.vocab_size, args.action_dim + 1),
        hidden_size=max(args.hidden_size * 2, args.hidden_size),
        n_layers=max(args.layers + 1, 2),
        n_heads=args.heads,
        n_kv_heads=args.heads,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=args.obs_dim,
        obs_tokens=args.obs_tokens,
        action_dim=args.action_dim,
        memory_slots=max(args.memory_slots, 2),
        dropout=0.0,
        use_world_pred=args.world_weight > 0.0,
        world_model_horizon=args.horizon,
        use_uncertainty=args.world_weight > 0.0,
    )
    teacher = VAGICore(teacher_cfg)
    load_checkpoint(teacher, optimizer=None, ckpt_path=args.teacher)
    teacher.eval()

    student_cfg = VAGIConfig(
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
        use_world_pred=args.world_weight > 0.0,
        world_model_horizon=args.horizon,
        use_uncertainty=args.world_weight > 0.0,
    )
    student = VAGICore(student_cfg)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    student.train()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        batch_iter = pack_batches(
            read_jsonl(args.data),
            batch_size=args.batch_size,
            horizon=args.horizon,
            gamma=args.gamma,
        )
        for batch in batch_iter:
            obs = batch["obs"]
            actions = batch["actions"]
            input_ids = actions.unsqueeze(1).clamp(max=student_cfg.vocab_size - 1)
            state = student.init_state(batch_size=obs.shape[0])

            with torch.no_grad():
                teacher_out = teacher.forward(input_ids=input_ids, obs=obs, state=None, return_loss=False)

            student_out = student.forward(input_ids=input_ids, obs=obs, state=state, return_loss=False)
            loss = _distill_loss(student_out["action_logits"], teacher_out["action_logits"], args.temperature)
            loss = loss + args.value_weight * F.mse_loss(student_out["value"], teacher_out["value"])

            if student_out["world_pred"] is not None and teacher_out["world_pred"] is not None:
                loss = loss + args.world_weight * F.mse_loss(
                    student_out["world_pred"], teacher_out["world_pred"].detach()
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            if step % 25 == 0:
                print(f"step={step} loss={loss.item():.6f} lr={get_lr(optimizer):.6f}")

    save_checkpoint(
        student,
        optimizer,
        step=step,
        out_dir=out_dir,
        extra={"epochs": args.epochs, "timestamp": time.time()},
    )


if __name__ == "__main__":
    main()
