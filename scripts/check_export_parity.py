"""Check parity between PyTorch, ONNX, and quantized ONNX exports."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from scripts.export_utils import build_metadata, load_metadata, meta_path_for, write_metadata
from scripts.quantize_onnx import main as quantize_main
from vagi_core import VAGIConfig, VAGICore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check PyTorch/ONNX parity.")
    parser.add_argument("--out-dir", type=str, default="exports/parity")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--obs-tokens", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--memory-slots", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--quant-mode", type=str, default="int8", choices=["int8", "uint8", "fp16"])
    return parser.parse_args()


def _build_model(args: argparse.Namespace) -> VAGICore:
    cfg = VAGIConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.heads,
        mlp_ratio=2.0,
        max_seq_len=max(args.seq_len, args.obs_tokens + args.seq_len + 4),
        obs_dim=args.obs_dim,
        obs_tokens=args.obs_tokens,
        action_dim=args.action_dim,
        memory_slots=args.memory_slots,
        dropout=0.0,
        use_world_pred=False,
    )
    model = VAGICore(cfg)
    if args.checkpoint:
        from io.checkpoint import load_checkpoint

        load_checkpoint(model, optimizer=None, ckpt_path=args.checkpoint)
    model.eval()
    return model


def _torch_forward(model: VAGICore, input_ids: torch.Tensor, obs: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        out = model.forward(input_ids=input_ids, obs=obs, state=None, return_loss=False)
        text_logits = out["text_logits"].detach().cpu().numpy()
        action_logits = out["action_logits"].detach().cpu().numpy()
        value = out["value"].detach().cpu().numpy()
    return text_logits, action_logits, value


def _onnx_forward(session, input_ids: np.ndarray, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs = session.run(None, {"input_ids": input_ids, "obs": obs})
    return outputs[0], outputs[1], outputs[2]


def _measure_latency(fn, runs: int) -> Tuple[float, float]:
    tracemalloc.start()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed / max(runs, 1), float(peak) / (1024.0 * 1024.0)


def main() -> None:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("onnxruntime is required for parity checks. Install via `pip install onnxruntime`.") from exc

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(args)
    input_ids = torch.zeros((args.batch_size, args.seq_len), dtype=torch.long)
    obs = torch.zeros((args.batch_size, args.obs_dim), dtype=torch.float32)

    onnx_path = out_dir / "vagi.onnx"
    from scripts.export_onnx import VAGIOnnxWrapper

    wrapper = VAGIOnnxWrapper(model).eval()
    torch.onnx.export(
        wrapper,
        (input_ids, obs),
        onnx_path.as_posix(),
        input_names=["input_ids", "obs"],
        output_names=["text_logits", "action_logits", "value"],
        dynamic_axes=None,
        dynamo=False,
        opset_version=17,
        do_constant_folding=True,
    )
    meta = build_metadata(cfg=model.cfg, export_format="onnx")
    write_metadata(onnx_path, meta)

    quant_path = out_dir / f"vagi.{args.quant_mode}.onnx"
    quant_args = [
        "scripts.quantize_onnx",
        "--input",
        onnx_path.as_posix(),
        "--output",
        quant_path.as_posix(),
        "--mode",
        args.quant_mode,
        "--disable-shape-infer",
    ]
    import sys

    sys.argv = quant_args
    quantize_main()

    torch_out = _torch_forward(model, input_ids, obs)
    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    quant_session = ort.InferenceSession(quant_path.as_posix(), providers=["CPUExecutionProvider"])
    onnx_out = _onnx_forward(session, input_ids.numpy(), obs.numpy())
    quant_out = _onnx_forward(quant_session, input_ids.numpy(), obs.numpy())

    def _compare(ref, other) -> bool:
        return bool(np.allclose(ref, other, atol=args.atol, rtol=args.rtol))

    parity = {
        "torch_vs_onnx": [_compare(t, o) for t, o in zip(torch_out, onnx_out)],
        "torch_vs_quant": [_compare(t, q) for t, q in zip(torch_out, quant_out)],
    }

    torch_latency, torch_mem = _measure_latency(lambda: _torch_forward(model, input_ids, obs), args.runs)
    onnx_latency, onnx_mem = _measure_latency(lambda: _onnx_forward(session, input_ids.numpy(), obs.numpy()), args.runs)
    quant_latency, quant_mem = _measure_latency(lambda: _onnx_forward(quant_session, input_ids.numpy(), obs.numpy()), args.runs)

    report = {
        "parity": parity,
        "atol": args.atol,
        "rtol": args.rtol,
        "latency_ms": {
            "torch": torch_latency * 1000.0,
            "onnx": onnx_latency * 1000.0,
            "quantized": quant_latency * 1000.0,
        },
        "peak_mem_mb": {
            "torch": torch_mem,
            "onnx": onnx_mem,
            "quantized": quant_mem,
        },
        "paths": {
            "onnx": str(onnx_path),
            "quantized": str(quant_path),
            "onnx_meta": str(meta_path_for(onnx_path)),
            "quant_meta": str(meta_path_for(quant_path)),
        },
    }
    report_path = out_dir / "parity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
