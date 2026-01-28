"""Train vAGI-core on a tensor dataset or random synthetic data."""

from __future__ import annotations

import argparse
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from vagi_core import VAGIConfig, VAGICore

from scripts.data_utils import RandomDataset, load_tensor_dataset, move_batch_to_device, validate_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vAGI-core.")
    parser.add_argument("--data", type=str, default=None, help="Path to a torch-saved dict dataset (.pt)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--with-obs", action="store_true", help="Use obs inputs")
    parser.add_argument("--with-world", action="store_true", help="Enable world prediction head/loss")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save", type=str, default=None, help="Path to save model checkpoint (.pt)")
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
        num_samples = args.steps_per_epoch * args.batch_size
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
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = build_config(args)
    model = VAGICore(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    loader = build_dataloader(args, cfg)
    global_step = 0

    model.train()
    for epoch in range(args.epochs):
        for step_idx, batch in enumerate(loader):
            if args.data is None and step_idx >= args.steps_per_epoch:
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
            loss = out["loss"]
            if loss is None:
                raise ValueError("No loss computed. Ensure labels/targets are provided.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % args.log_every == 0:
                print(f"epoch={epoch + 1} step={global_step} loss={loss.item():.6f}")

    if args.save:
        payload = {"model_state": model.state_dict(), "config": cfg.__dict__}
        torch.save(payload, args.save)
        print(f"Saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
