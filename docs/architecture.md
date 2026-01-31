# ARCHITECTURE

This document describes the vAGI core model and its minimal training/runtime stack.

## vAGI-core
- Single causal transformer decoder (pre-norm).
- Optional observation tokens prepended to text tokens.
- Multi-head outputs: language logits, policy logits, value, and optional world prediction.
- Optional vision encoder projects image observations into `obs_dim` vectors before tokenization.

## State and memory
- `RecurrentState` tracks:
  - `mem`: fast memory slots `(B, M, D)`.
  - `kv`: per-layer key/value cache (stubbed but typed for future extension).
  - `timestep`: monotonically increasing integer.
- `FastMemory` writes from the last hidden state using a gated additive update.

## Heads
- `LanguageHead`: token-level distribution over the vocabulary.
- `PolicyHead`: action logits for discrete action spaces.
- `ValueHead`: scalar value prediction.
- `WorldHead`: optional next-observation prediction.

## Cache
- `KVCache` in `core/memory.py` stores per-layer keys/values for step-wise decoding.
- Optional `max_len` truncates cached history for bounded memory usage.

## Training loops
- `scripts/train.py`: minimal supervised language-model training.
- `scripts/train_policy_bc.py`: behavior cloning on action types.
- `scripts/train_value.py`: value head regression on returns.
- `scripts/train_world_model.py`: world model regression on next observations.

## Inference tuning
- Batch size can be tuned for throughput in scripts and benchmarks.
- `scripts/bench_latency.py` supports a `--compile` flag for `torch.compile`.
- `scripts/tune_inference.py` sweeps batch size and optional KV cache lengths.

See `README.md` for diagrams.
