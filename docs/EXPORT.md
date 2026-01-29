# EXPORT

## ONNX
```bash
python -m scripts.export_onnx --out exports/vagi.onnx
```

Notes:
- Requires `onnx` package installed separately.
- Exported graph includes text logits, action logits, and value outputs.

## TensorRT (optional)
- Convert the ONNX export using your TensorRT toolchain.
- TensorRT integration is not bundled in this repo to keep dependencies minimal.
