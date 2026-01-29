# ARCHITECTURE

This document describes the vAGI core model and its minimal training/runtime stack.

## vAGI-core
- Single causal transformer decoder (pre-norm).
- Optional observation tokens prepended to text tokens.
- Multi-head outputs: language logits, policy logits, value, and optional world prediction.

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
- `KVCache` in `core/memory.py` is a placeholder container to allow future KV
  implementations without refactoring the public API.

## Training loops
- `scripts/train.py`: minimal supervised language-model training.
- `scripts/train_policy_bc.py`: behavior cloning on action types.
- `scripts/train_value.py`: value head regression on returns.
- `scripts/train_world_model.py`: world model regression on next observations.

## Inference tuning
- Batch size can be tuned for throughput in scripts and benchmarks.
- `scripts/bench_latency.py` supports a `--compile` flag for `torch.compile`.
- KV-cache settings are currently stubbed; future tuning should replace `KVCache`.
