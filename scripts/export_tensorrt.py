"""Export an ONNX model to TensorRT (optional dependency)."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX.")
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model.")
    parser.add_argument("--out", type=str, default="exports/vagi.trt")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--workspace-mb", type=int, default=512)
    return parser.parse_args()


def _build_engine(args: argparse.Namespace) -> bytes:
    try:
        import tensorrt as trt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "TensorRT is required for this export. Install TensorRT or use ONNX export only."
        ) from exc

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX file: {onnx_path}")
    with onnx_path.open("rb") as handle:
        if not parser.parse(handle.read()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"Failed to parse ONNX: {errors}")

    config = builder.create_builder_config()
    config.max_workspace_size = int(args.workspace_mb) * 1024 * 1024
    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", (args.batch_size, args.seq_len), (args.batch_size, args.seq_len), (args.batch_size, args.seq_len))
    profile.set_shape("obs", (args.batch_size, args.obs_dim), (args.batch_size, args.obs_dim), (args.batch_size, args.obs_dim))
    config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    return engine.serialize()


def main() -> None:
    args = parse_args()
    engine_data = _build_engine(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(engine_data)
    print(f"Wrote TensorRT engine to {out_path}")


if __name__ == "__main__":
    main()
