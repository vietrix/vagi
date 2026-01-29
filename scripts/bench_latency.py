"""Benchmark VAGICore step latency with optional torch.compile."""

from __future__ import annotations

import argparse
import time

import torch

from vagi_core import VAGIConfig, VAGICore
from scripts.utils import set_deterministic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark VAGICore step latency.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--kv-cache-len", type=int, default=None)
    parser.add_argument("--prefill-kv", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed, args.deterministic)

    cfg = VAGIConfig(
        vocab_size=128,
        hidden_size=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=args.obs_dim,
        obs_tokens=2,
        action_dim=8,
        memory_slots=4,
        dropout=0.0,
        use_world_pred=False,
    )
    model = VAGICore(cfg)
    model.eval()

    if args.compile:
        if hasattr(torch, "compile"):
            model = torch.compile(model)
        else:
            print("torch.compile not available; continuing without compile.")

    input_ids = torch.zeros((args.batch_size, args.seq_len), dtype=torch.long)
    obs = torch.zeros((args.batch_size, args.obs_dim), dtype=torch.float32)
    state = model.init_state(
        batch_size=args.batch_size,
        prefill_kv=args.prefill_kv,
        kv_max_seq_len=args.kv_cache_len,
    )

    for _ in range(args.warmup):
        _ = model.step(input_ids=input_ids, obs=obs, state=state)

    start = time.perf_counter()
    for _ in range(args.steps):
        _ = model.step(input_ids=input_ids, obs=obs, state=state)
    elapsed = time.perf_counter() - start
    per_step_ms = (elapsed / max(args.steps, 1)) * 1000.0
    print(f"avg_step_ms={per_step_ms:.3f} (steps={args.steps})")


if __name__ == "__main__":
    main()
