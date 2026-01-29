"""Offline advantage-weighted behavior cloning from JSONL rollouts."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from io.checkpoint import save_checkpoint
from scripts.utils import get_lr, set_deterministic
from utils.data.pack import pack_batches
from utils.data.reader import read_jsonl
from vagi_core import VAGIConfig, VAGICore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train policy head from offline rollouts (AWBC).")
    parser.add_argument("--data", type=str, default="logs/rollouts.jsonl")
    parser.add_argument("--out-dir", type=str, default="runs/offline_policy")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--weight-clip", type=float, default=20.0)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--anchor", type=str, default=None)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
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
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=100)
    return parser.parse_args()


def _load_anchor(path: Optional[str]) -> Optional[Dict[str, torch.Tensor]]:
    if not path:
        return None
    return load_file(str(path))


def _anchor_loss(model: torch.nn.Module, anchor_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    loss = None
    for name, param in model.named_parameters():
        if name not in anchor_state:
            continue
        anchor = anchor_state[name].to(param.device)
        delta = torch.mean((param - anchor) ** 2)
        loss = delta if loss is None else loss + delta
    if loss is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return loss


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed, args.deterministic)

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
        use_world_pred=False,
    )
    model = VAGICore(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    anchor_state = _load_anchor(args.anchor)

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
            returns = batch["returns"]
            if int(actions.max().item()) >= cfg.action_dim:
                raise ValueError("action id exceeds action_dim")

            input_ids = actions.unsqueeze(1).clamp(max=cfg.vocab_size - 1)
            state = model.init_state(batch_size=obs.shape[0])
            out = model.forward(input_ids=input_ids, obs=obs, state=state, return_loss=False)

            logits = out["action_logits"]
            values = out["value"]
            advantages = returns - values.detach()
            adv_scaled = torch.clamp(advantages / max(args.temperature, 1e-6), min=-20.0, max=20.0)
            weights = torch.exp(adv_scaled).clamp(max=args.weight_clip).squeeze(-1)

            ce = F.cross_entropy(logits, actions, reduction="none")
            policy_loss = torch.mean(ce * weights)
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + args.value_weight * value_loss

            if anchor_state is not None and args.anchor_weight > 0.0:
                loss = loss + args.anchor_weight * _anchor_loss(model, anchor_state)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            if args.log_every and step % args.log_every == 0:
                print(
                    "step={step} loss={loss:.6f} lr={lr:.6f}".format(
                        step=step, loss=loss.item(), lr=get_lr(optimizer)
                    )
                )
            if args.save_every and step % args.save_every == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    step=step,
                    out_dir=out_dir,
                    extra={"epoch": epoch + 1, "timestamp": time.time()},
                )

    save_checkpoint(
        model,
        optimizer,
        step=step,
        out_dir=out_dir,
        extra={"epoch": args.epochs, "timestamp": time.time()},
    )


if __name__ == "__main__":
    main()
