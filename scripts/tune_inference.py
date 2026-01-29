"""Sweep batch size and KV cache settings for inference timing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import torch

from vagi_core import VAGIConfig, VAGICore
from scripts.utils import set_deterministic


def _parse_list(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune inference batch/KV cache.")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8")
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--kv-cache-len", type=int, default=None)
    parser.add_argument("--prefill-kv", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--out", type=str, default="results/inference_tuning.json")
    return parser.parse_args()


def _bench(model: VAGICore, batch_size: int, seq_len: int, obs_dim: int, steps: int, warmup: int, kv_len: int | None, prefill_kv: bool) -> float:
    input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    obs = torch.zeros((batch_size, obs_dim), dtype=torch.float32)
    state = model.init_state(
        batch_size=batch_size,
        prefill_kv=prefill_kv,
        kv_max_seq_len=kv_len,
    )
    for _ in range(warmup):
        _ = model.step(input_ids=input_ids, obs=obs, state=state)
    start = time.perf_counter()
    for _ in range(steps):
        _ = model.step(input_ids=input_ids, obs=obs, state=state)
    elapsed = time.perf_counter() - start
    return (elapsed / max(steps, 1)) * 1000.0


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed, args.deterministic)

    batch_sizes = _parse_list(args.batch_sizes)
    results = []

    for batch_size in batch_sizes:
        cfg = VAGIConfig(
            vocab_size=128,
            hidden_size=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=4,
            mlp_ratio=2.0,
            max_seq_len=max(args.seq_len, args.kv_cache_len or args.seq_len),
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

        avg_ms = _bench(
            model=model,
            batch_size=batch_size,
            seq_len=args.seq_len,
            obs_dim=args.obs_dim,
            steps=args.steps,
            warmup=args.warmup,
            kv_len=args.kv_cache_len,
            prefill_kv=args.prefill_kv,
        )
        results.append(
            {
                "batch_size": batch_size,
                "seq_len": args.seq_len,
                "obs_dim": args.obs_dim,
                "avg_step_ms": avg_ms,
                "compile": bool(args.compile),
                "kv_cache_len": args.kv_cache_len,
                "prefill_kv": bool(args.prefill_kv),
            }
        )
        print(f"batch={batch_size} avg_step_ms={avg_ms:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
