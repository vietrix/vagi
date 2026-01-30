"""Quantize an ONNX export for lightweight deployment."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize ONNX model using onnxruntime.")
    parser.add_argument("--input", type=str, required=True, help="Path to input ONNX model.")
    parser.add_argument("--output", type=str, required=True, help="Path to output quantized ONNX model.")
    parser.add_argument("--weight-type", type=str, default="int8", choices=["int8", "uint8"])
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--reduce-range", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "onnxruntime is required for quantization. Install via `pip install onnxruntime`."
        ) from exc

    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    weight_type = QuantType.QInt8 if args.weight_type == "int8" else QuantType.QUInt8
    quantize_dynamic(
        model_input=input_path.as_posix(),
        model_output=output_path.as_posix(),
        weight_type=weight_type,
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
    )
    print(f"Wrote quantized model to {output_path}")


if __name__ == "__main__":
    main()
