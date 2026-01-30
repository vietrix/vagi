# EXPORT

## ONNX
```bash
python -m scripts.export_onnx --out exports/vagi.onnx
```

Notes:
- Requires `onnx` package installed separately.
- Exported graph includes text logits, action logits, and value outputs.
- A metadata file is written alongside the export: `exports/vagi.onnx.meta.json`.

## Artifact naming convention
Recommended export names:

- `exports/vagi_full.onnx`
- `exports/vagi_full.int8.onnx`
- `exports/vagi_small.onnx`
- `exports/vagi_small.int8.onnx`

Each export writes a metadata file `{artifact}.meta.json` with:

- `schema_version`
- `git_sha`
- `config_hash`
- `config` (full model config)
- `export` info (format, quantization, source)

## Manifest
The repository tracks `exports/manifest.json` with:

- model name/version
- git SHA
- config hash
- schema version
- backend targets (onnx/tensorrt)
- expected metrics from golden runs

## Parity + latency check
```bash
python -m scripts.check_export_parity --quant-mode int8 --runs 50
```

Outputs:
- `exports/parity/parity_report.json`

## ONNX quantization (optional)
```bash
python -m scripts.quantize_onnx --input exports/vagi.onnx --output exports/vagi.int8.onnx
```

Notes:
- Requires `onnxruntime` package installed separately.
- Uses dynamic weight-only quantization by default.
- For fp16 conversion: `python -m scripts.quantize_onnx --input exports/vagi.onnx --output exports/vagi.fp16.onnx --mode fp16`

## TensorRT (optional)
```bash
python -m scripts.export_tensorrt --onnx exports/vagi.onnx --out exports/vagi.trt
```

Notes:
- Requires NVIDIA TensorRT installed locally.
- Uses a fixed optimization profile from `--batch-size`, `--seq-len`, and `--obs-dim`.
