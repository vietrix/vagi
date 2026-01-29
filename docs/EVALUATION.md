# EVALUATION

This document describes the minimal evaluation setup for vAGI.

## Benchmarks
### CodeEnv regression tasks
- Location: `envs/code_env/fixtures/benchmarks`
- 20 deterministic tasks with known fixes.
- Curriculum levels are defined in `envs/code_env/fixtures/benchmarks/manifest.json`.

### Runner
- Single run: `python -m scripts.bench_code_env`
- Multi-seed reproducibility: `python -m scripts.run_all_benchmarks`

## Metrics
- `pass_rate`: fraction of tasks with zero failing tests.
- `avg_steps`: mean action steps per task.
- `avg_runs`: mean `run_tests` invocations.
- `avg_time`: mean wall time per task.

## Seeds
- Default seeds are `0..9` in `scripts/run_all_benchmarks.py`.
- Use `--deterministic` for deterministic PyTorch kernels.

## Hardware
- CPU-only by default.
- GPU is not required for the minimal evaluation suite.
