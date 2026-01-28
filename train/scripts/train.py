"""Minimal language-model training script for vAGI-core."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from vagi_core import VAGIConfig, VAGICore

from train.scripts.collate import make_collate_fn
from train.scripts.config import TrainConfig
from train.scripts.dataset_text import TextDataset, build_tokenizer, load_texts
from io.checkpoint import load_checkpoint, save_checkpoint
from train.scripts.utils import get_lr, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vAGI-core on a tiny text dataset.")
    parser.add_argument("--data", type=str, default="data/sample/sample.txt")
    parser.add_argument("--out-dir", type=str, default="runs/minimal")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--obs-dim", type=int, default=0)
    parser.add_argument("--obs-tokens", type=int, default=0)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--memory-slots", type=int, default=4)
    return parser.parse_args()


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        data_path=args.data,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        save_every=args.save_every,
        log_every=args.log_every,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        hidden_size=args.hidden_size,
        n_layers=args.layers,
        n_heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        obs_dim=args.obs_dim,
        obs_tokens=args.obs_tokens,
        action_dim=args.action_dim,
        memory_slots=args.memory_slots,
        use_world_pred=False,
    )


def main() -> None:
    args = parse_args()
    cfg = build_train_config(args)
    set_seed(cfg.seed)

    texts = load_texts(cfg.data_path)
    tokenizer = build_tokenizer(texts)
    dataset = TextDataset(texts, tokenizer, max_length=cfg.max_seq_len)

    collate_fn = make_collate_fn(
        pad_id=tokenizer.pad_id,
        max_length=cfg.max_seq_len,
        obs_dim=cfg.obs_dim,
        add_obs=cfg.obs_dim > 0,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)

    model_cfg = VAGIConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=cfg.hidden_size,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_heads,
        mlp_ratio=cfg.mlp_ratio,
        max_seq_len=cfg.max_seq_len,
        obs_dim=max(cfg.obs_dim, 1) if cfg.obs_dim > 0 else 1,
        obs_tokens=cfg.obs_tokens,
        action_dim=cfg.action_dim,
        memory_slots=cfg.memory_slots,
        dropout=0.0,
        use_world_pred=False,
    )

    model = VAGICore(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model.train()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    if args.resume:
        meta = load_checkpoint(model, optimizer=optimizer, ckpt_path=args.resume)
        step = int(meta.get("step", 0))
    for epoch in range(cfg.epochs):
        for batch in loader:
            step += 1
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            obs = batch.get("obs") if cfg.obs_dim > 0 else None
            state = model.init_state(input_ids.shape[0])

            out = model.forward(input_ids=input_ids, obs=obs, state=state, labels=labels, return_loss=True)
            loss = out["loss"]
            if loss is None:
                raise ValueError("No loss returned by model.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % cfg.log_every == 0:
                print(f"step={step} loss={loss.item():.6f} lr={get_lr(optimizer):.6f}")

            if cfg.save_every and step % cfg.save_every == 0:
                save_checkpoint(model, optimizer, step=step, out_dir=out_dir, extra={"epoch": epoch + 1})

            if cfg.max_steps is not None and step >= cfg.max_steps:
                break
        if cfg.max_steps is not None and step >= cfg.max_steps:
            break


if __name__ == "__main__":
    main()
