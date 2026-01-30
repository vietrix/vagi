"""Profile inference stages and report timing breakdown."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from vagi_core import VAGIConfig, VAGICore
from vagi_core.utils import StageTimer
from io.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile inference stages.")
    parser.add_argument("--backend", type=str, default="pytorch", choices=["pytorch", "onnxruntime", "tensorrt"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--onnx", type=str, default=None)
    parser.add_argument("--tensorrt", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", type=str, default="results/profile_infer.json")
    return parser.parse_args()


def _build_model(args: argparse.Namespace) -> VAGICore:
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
        use_world_pred=True,
    )
    model = VAGICore(cfg)
    if args.checkpoint:
        load_checkpoint(model, optimizer=None, ckpt_path=args.checkpoint)
    model.eval()
    return model


def _profile_pytorch(args: argparse.Namespace) -> Dict[str, float]:
    model = _build_model(args)
    timer = StageTimer()
    input_ids = torch.zeros((args.batch_size, args.seq_len), dtype=torch.long)
    obs = torch.zeros((args.batch_size, args.obs_dim), dtype=torch.float32)
    state = model.init_state(batch_size=args.batch_size)

    for _ in range(args.warmup):
        _ = model.step(input_ids=input_ids, obs=obs, state=state, timer=timer)

    timer.times.clear()
    start = time.perf_counter()
    for _ in range(args.steps):
        _ = model.step(input_ids=input_ids, obs=obs, state=state, timer=timer)
    elapsed = time.perf_counter() - start

    total_ms = (elapsed / max(args.steps, 1)) * 1000.0
    stage_ms = {name: (value / max(args.steps, 1)) * 1000.0 for name, value in timer.times.items()}
    return {"total_ms": total_ms, "stages_ms": stage_ms}


def _profile_onnx(args: argparse.Namespace) -> Dict[str, float]:
    if not args.onnx:
        raise ValueError("--onnx is required for onnxruntime profiling")
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("onnxruntime is required for onnx profiling.") from exc
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_ids = np.zeros((args.batch_size, args.seq_len), dtype=np.int64)
    obs = np.zeros((args.batch_size, args.obs_dim), dtype=np.float32)

    for _ in range(args.warmup):
        _ = session.run(None, {"input_ids": input_ids, "obs": obs})
    start = time.perf_counter()
    for _ in range(args.steps):
        _ = session.run(None, {"input_ids": input_ids, "obs": obs})
    elapsed = time.perf_counter() - start
    total_ms = (elapsed / max(args.steps, 1)) * 1000.0
    return {"total_ms": total_ms, "stages_ms": {}}


def _profile_tensorrt(args: argparse.Namespace) -> Dict[str, float]:
    _ = args  # unused for now
    raise NotImplementedError("TensorRT profiling requires a runtime integration.")


def main() -> None:
    args = parse_args()
    if args.backend == "pytorch":
        result = _profile_pytorch(args)
    elif args.backend == "onnxruntime":
        result = _profile_onnx(args)
    else:
        result = _profile_tensorrt(args)

    payload = {
        "backend": args.backend,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "obs_dim": args.obs_dim,
        "steps": args.steps,
        "warmup": args.warmup,
        "total_ms_per_step": result["total_ms"],
        "stage_ms_per_step": result["stages_ms"],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
