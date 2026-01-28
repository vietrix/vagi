"""Run step-wise inference and print state updates."""

from __future__ import annotations

import argparse

import torch

from vagi_core import VAGIConfig, VAGICore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vAGI-core step loop.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--obs-tokens", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--memory-slots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = VAGIConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.heads,
        mlp_ratio=2.0,
        max_seq_len=max(args.tokens, 8),
        obs_dim=args.obs_dim,
        obs_tokens=args.obs_tokens,
        action_dim=args.action_dim,
        memory_slots=args.memory_slots,
        dropout=0.0,
        use_world_pred=False,
    )
    model = VAGICore(cfg).to(device)
    model.eval()

    state = model.init_state(args.batch, device=device)
    obs = torch.randn(args.batch, cfg.obs_dim, device=device)

    for step_idx in range(args.steps):
        token = torch.randint(0, cfg.vocab_size, (args.batch, args.tokens), dtype=torch.long, device=device)
        out = model.step(input_ids=token, obs=obs, state=state)
        state = out["state"]
        mem_norm = state.mem.norm().item()
        print(f"step={step_idx + 1} timestep={state.timestep} mem_norm={mem_norm:.4f}")


if __name__ == "__main__":
    main()
