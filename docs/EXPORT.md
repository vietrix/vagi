# EXPORT

## ONNX
```bash
python -m scripts.export_onnx --out exports/vagi.onnx
```

Notes:
- Requires `onnx` package installed separately.
- Exported graph includes text logits, action logits, and value outputs.

## ONNX quantization (optional)
```bash
python -m scripts.quantize_onnx --input exports/vagi.onnx --output exports/vagi.int8.onnx
```

Notes:
- Requires `onnxruntime` package installed separately.
- Uses dynamic weight-only quantization by default.

## TensorRT (optional)
```bash
python -m scripts.export_tensorrt --onnx exports/vagi.onnx --out exports/vagi.trt
```

Notes:
- Requires NVIDIA TensorRT installed locally.
- Uses a fixed optimization profile from `--batch-size`, `--seq-len`, and `--obs-dim`.
