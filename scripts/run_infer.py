"""Run inference on selectable backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from vagi_core import VAGIConfig, VAGICore
from io.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with selectable backend.")
    parser.add_argument("--backend", type=str, default="pytorch", choices=["pytorch", "onnxruntime", "tensorrt"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--onnx", type=str, default=None)
    parser.add_argument("--tensorrt", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config for VAGIConfig.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--obs-tokens", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--memory-slots", type=int, default=4)
    parser.add_argument("--use-world", action="store_true")
    parser.add_argument("--use-uncertainty", action="store_true")
    parser.add_argument("--mode", type=str, default="forward", choices=["forward", "step", "plan"])
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--uncertainty-fallback", type=float, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> VAGIConfig:
    if args.config:
        payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
        return VAGIConfig(**payload)
    return VAGIConfig(
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
        use_world_pred=args.use_world,
        use_uncertainty=args.use_uncertainty,
    )


def _summarize(outputs: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key in ("text_logits", "action_logits", "value", "world_pred"):
        value = outputs.get(key)
        if value is None:
            summary[key] = None
        elif isinstance(value, np.ndarray):
            summary[key] = {"shape": list(value.shape), "mean": float(value.mean())}
        elif torch.is_tensor(value):
            summary[key] = {"shape": list(value.shape), "mean": float(value.mean().item())}
        else:
            summary[key] = str(type(value))
    action_logits = outputs.get("action_logits")
    if action_logits is not None:
        logits = action_logits if isinstance(action_logits, np.ndarray) else action_logits.detach().cpu().numpy()
        summary["action"] = int(np.argmax(logits, axis=-1).flatten()[0])
    return summary


def _run_pytorch(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _build_config(args)
    model = VAGICore(cfg)
    if args.checkpoint:
        load_checkpoint(model, optimizer=None, ckpt_path=args.checkpoint)
    model.eval()
    input_ids = torch.zeros((args.batch_size, args.seq_len), dtype=torch.long)
    obs = torch.zeros((args.batch_size, args.obs_dim), dtype=torch.float32)
    if args.mode == "forward":
        out = model.forward(input_ids=input_ids, obs=obs, state=None, return_loss=False)
        return out
    state = model.init_state(batch_size=args.batch_size)
    if args.mode == "step":
        return model.step(input_ids=input_ids, obs=obs, state=state)
    plan = model.plan_step(
        input_ids=input_ids,
        obs=obs,
        state=state,
        horizon=args.horizon,
        num_candidates=args.num_candidates,
        uncertainty_fallback=args.uncertainty_fallback,
    )
    return {
        "text_logits": None,
        "action_logits": plan.get("action_logits"),
        "value": plan.get("candidate_values"),
        "world_pred": None,
        "action": plan.get("action"),
    }


def _run_onnx(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.onnx:
        raise ValueError("--onnx is required for onnxruntime backend")
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("onnxruntime is required for onnxruntime backend.") from exc
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_ids = np.zeros((args.batch_size, args.seq_len), dtype=np.int64)
    obs = np.zeros((args.batch_size, args.obs_dim), dtype=np.float32)
    outputs = session.run(None, {"input_ids": input_ids, "obs": obs})
    names = [out.name for out in session.get_outputs()]
    return {name: value for name, value in zip(names, outputs)}


def _run_tensorrt(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.tensorrt:
        raise ValueError("--tensorrt is required for tensorrt backend")
    try:
        import tensorrt as trt  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("TensorRT is required for tensorrt backend.") from exc
    raise NotImplementedError("TensorRT inference requires a dedicated runtime integration.")


def main() -> None:
    args = parse_args()
    if args.backend == "pytorch":
        outputs = _run_pytorch(args)
    elif args.backend == "onnxruntime":
        outputs = _run_onnx(args)
    else:
        outputs = _run_tensorrt(args)

    summary = _summarize(outputs)
    summary["backend"] = args.backend
    summary["mode"] = args.mode
    text = json.dumps(summary, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
