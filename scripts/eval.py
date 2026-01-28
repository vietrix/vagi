"""Evaluate vAGI-core losses on a dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from vagi_core import VAGIConfig, VAGICore

from scripts.checkpoint import load_checkpoint, load_config_from_checkpoint
from scripts.data_utils import RandomDataset, load_tensor_dataset, move_batch_to_device, validate_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate vAGI-core.")
    parser.add_argument("--data", type=str, default=None, help="Path to a torch-saved dict dataset (.pt)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--with-obs", action="store_true")
    parser.add_argument("--with-world", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint directory or .safetensors file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--obs-tokens", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--memory-slots", type=int, default=4)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> VAGIConfig:
    return VAGIConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.heads,
        mlp_ratio=2.0,
        max_seq_len=max(args.seq_len, 8),
        obs_dim=args.obs_dim,
        obs_tokens=args.obs_tokens,
        action_dim=args.action_dim,
        memory_slots=args.memory_slots,
        dropout=0.0,
        use_world_pred=args.with_world,
    )


def build_dataloader(args: argparse.Namespace, cfg: VAGIConfig) -> DataLoader:
    if args.data:
        dataset = load_tensor_dataset(args.data)
    else:
        num_samples = args.steps * args.batch_size
        dataset = RandomDataset(
            num_samples=num_samples,
            seq_len=args.seq_len,
            vocab_size=cfg.vocab_size,
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            with_obs=args.with_obs,
            with_world=args.with_world,
            seed=args.seed,
        )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = load_config_from_checkpoint(args.checkpoint) if args.checkpoint else None
    if cfg is None:
        cfg = build_config(args)
    model = VAGICore(cfg).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model=model, optimizer=None, device=device)
    model.eval()

    loader = build_dataloader(args, cfg)
    loss_total = 0.0
    count = 0

    with torch.no_grad():
        for step_idx, batch in enumerate(loader):
            if args.data is None and step_idx >= args.steps:
                break
            batch = move_batch_to_device(batch, device)
            validate_batch(batch, cfg, require_obs=args.with_obs)

            input_ids = batch["input_ids"]
            labels = batch.get("labels", input_ids)
            obs = batch.get("obs") if args.with_obs else None
            state = model.init_state(input_ids.shape[0], device=device)

            targets: Dict[str, torch.Tensor] = {}
            if "actions" in batch:
                targets["actions"] = batch["actions"]
            if "values" in batch:
                targets["values"] = batch["values"]
            if args.with_world and "obs_next" in batch:
                targets["obs_next"] = batch["obs_next"]

            out = model.forward(
                input_ids=input_ids,
                obs=obs,
                state=state,
                labels=labels,
                targets=targets,
                return_loss=True,
            )
            if out["loss"] is None:
                continue
            loss_total += out["loss"].item()
            count += 1

    mean_loss = loss_total / max(count, 1)
    print(f"mean_loss={mean_loss:.6f} steps={count}")


if __name__ == "__main__":
    main()
