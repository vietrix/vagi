# vAGI Checkpoints

This repo uses **safetensors** for model weights and standard `torch.save` for optimizer state.

## File layout

When using `io/checkpoint.py`, a checkpoint directory contains:

- `model.safetensors` — model weights (required)
- `meta.json` — metadata (step, extra info)
- `optimizer.pt` — optimizer state (optional)

## Naming convention

Default filenames:

- `model.safetensors`
- `optimizer.pt`
- `meta.json`

You may store checkpoints under experiment folders like:

```
runs/offline_policy/
  model.safetensors
  optimizer.pt
  meta.json
```

## Anchor checkpoints (anti-forgetting)

Offline training scripts can apply L2-to-anchor regularization by passing a safetensors
checkpoint path:

- `--anchor path/to/model.safetensors`
- `--anchor-weight 0.01`

This keeps the model close to a stable anchor during distillation or replay.

## ONNX / TensorRT exports

ONNX/TensorRT exports are optional and should be created only when needed for deployment.
Use a separate output directory and record model version + config in metadata.

Export helpers write `{artifact}.meta.json` with schema version, git SHA, and config hash.
