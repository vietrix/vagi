"""Quantize an ONNX export for lightweight deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.export_utils import build_metadata, load_metadata, meta_path_for, write_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize ONNX model using onnxruntime.")
    parser.add_argument("--input", type=str, required=True, help="Path to input ONNX model.")
    parser.add_argument("--output", type=str, required=True, help="Path to output quantized ONNX model.")
    parser.add_argument("--mode", type=str, default=None, choices=["int8", "uint8", "fp16"])
    parser.add_argument("--weight-type", type=str, default="int8", choices=["int8", "uint8"])
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--reduce-range", action="store_true")
    parser.add_argument("--meta-in", type=str, default=None)
    parser.add_argument("--meta-out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = args.mode or args.weight_type
    if mode == "fp16":
        try:
            import onnx
            from onnx import numpy_helper
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("onnx is required for fp16 conversion. Install via `pip install onnx`.") from exc
        model = onnx.load(input_path.as_posix())
        for initializer in model.graph.initializer:
            if initializer.data_type == onnx.TensorProto.FLOAT:
                array = numpy_helper.to_array(initializer).astype(np.float16)
                initializer.ClearField("raw_data")
                initializer.raw_data = array.tobytes()
                initializer.data_type = onnx.TensorProto.FLOAT16
        onnx.save(model, output_path.as_posix())
    else:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "onnxruntime is required for int8 quantization. Install via `pip install onnxruntime`."
            ) from exc
        weight_type = QuantType.QInt8 if mode == "int8" else QuantType.QUInt8
        quantize_dynamic(
            model_input=input_path.as_posix(),
            model_output=output_path.as_posix(),
            weight_type=weight_type,
            per_channel=args.per_channel,
            reduce_range=args.reduce_range,
        )
    meta_in = Path(args.meta_in) if args.meta_in else meta_path_for(input_path)
    base_meta = load_metadata(meta_in) if meta_in.exists() else None
    meta = build_metadata(
        cfg=None,
        export_format="onnx",
        quantization=mode,
        source=str(input_path),
        base_meta=base_meta,
    )
    meta_out = Path(args.meta_out) if args.meta_out else output_path
    if args.meta_out and meta_out.suffix == ".json":
        meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        write_metadata(meta_out, meta)
    print(f"Wrote quantized model to {output_path}")


if __name__ == "__main__":
    main()
