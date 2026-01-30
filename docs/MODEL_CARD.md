# vAGI Model Card (Core)

## Overview

vAGI is a **model‑centric** architecture: one core model supports both fast actions and
deliberate planning. It is not an agent framework; tool use and runtime orchestration
are layered on top of the same neural core.

## Capabilities

- Unified policy/value/world modeling
- Reflection heads for error awareness and information gain
- Risk‑aware planning (CEM / tree search)
- Adaptive compute via budget controller
- Long‑horizon memory with stabilization

## Intended Use

Research and prototyping of unified decision‑making. The core is designed for offline
training and can be distilled to smaller models for efficiency.

## Performance & evaluation

This repository ships with **untrained defaults**; performance depends on the
training pipeline and data you run. Use the built‑in suites to generate numbers:

- Cross‑env generalization: `python -m scripts.bench_cross_env`
- Multi‑seed baseline suite: `python -m scripts.run_all_benchmarks`
- Self‑improve loop: `python -m scripts.self_improve_multi_env`
- Export parity + latency: `python -m scripts.check_export_parity`

Results are saved under `results/` and `exports/parity/` to support reporting.

## Limitations

- Planning quality depends on world model fidelity.
- Reflection heads require curated targets to be reliable.
- No external tool or environment guarantees are implied.
- Untrained checkpoints will not solve tasks without additional training.

## Known failure modes

- **Tool misuse**: selecting invalid or premature actions in the code environment.
- **Patch overreach**: applying diffs without sufficient evidence (can regress tests).
- **World drift**: multi‑step rollouts diverge when the world model is under‑trained.
- **Value miscalibration**: value estimates can be over‑confident without uncertainty calibration.
- **Vision shortcuts**: UI tasks can fail when the vision encoder is not trained for the domain.

## Model‑centric AGI rationale

vAGI keeps intelligence **inside a single model** with:

1. Shared representations for language, actions, value, and world prediction.
2. Two compute modes (act vs think) without model switching.
3. Self‑assessment heads to control uncertainty and exploration.

This keeps the system modular while preserving a unified "brain" for learning and
adaptation.
