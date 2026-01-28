"""Run a forward pass with supervised losses and print totals."""

from __future__ import annotations

import argparse

import torch

from vagi_core import VAGIConfig, VAGICore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vAGI-core loss computation.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=5)
    parser.add_argument("--with-obs", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = VAGIConfig(
        vocab_size=128,
        hidden_size=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=max(args.tokens, 8),
        obs_dim=16,
        obs_tokens=2,
        action_dim=8,
        memory_slots=4,
        dropout=0.0,
        use_world_pred=True,
    )
    model = VAGICore(cfg).to(device)
    model.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (args.batch, args.tokens), dtype=torch.long, device=device)
    obs = torch.randn(args.batch, cfg.obs_dim, device=device) if args.with_obs else None
    state = model.init_state(args.batch, device=device)

    labels = torch.randint(0, cfg.vocab_size, (args.batch, args.tokens), dtype=torch.long, device=device)
    targets = {
        "actions": torch.randint(0, cfg.action_dim, (args.batch,), dtype=torch.long, device=device),
        "values": torch.randn(args.batch, 1, device=device),
        "obs_next": torch.randn(args.batch, cfg.obs_dim, device=device),
        "loss_weights": {"language": 1.0, "policy": 1.0, "value": 0.5, "world": 0.25},
    }

    out = model.forward(
        input_ids=input_ids,
        obs=obs,
        state=state,
        labels=labels,
        targets=targets,
        return_loss=True,
    )
    print("loss:", out["loss"].item())
    for name, val in out["losses_breakdown"].items():
        print(f"{name}: {val.item():.6f}")


if __name__ == "__main__":
    main()
