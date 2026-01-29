# EVALUATION

This document describes the evaluation setup for vAGI and baseline agents.

## Benchmarks
### CodeEnv regression tasks
- Location: `envs/code_env/fixtures/benchmarks`
- 20 deterministic tasks with known fixes.
- Curriculum levels are defined in `envs/code_env/fixtures/benchmarks/manifest.json`.

### Runner
- Single run: `python -m scripts.bench_code_env`
- Multi-seed reproducibility: `python -m scripts.run_all_benchmarks`
- Baselines: `python -m scripts.eval_baselines`

## Baselines
### Random baseline
- Chooses actions uniformly from the action space.
- Uses fixed valid arguments for file operations.

### Heuristic baseline
- Deterministic rule based on normalized observation features.
- Cycles through planning and inspection before verifying.

### LLM stub baseline
- Offline fake LLM that maps observation features to an action type deterministically.
- Used for local testing without network access.

### LLM OpenAI baseline (optional)
- Runs only when `OPENAI_API_KEY` is set.
- Uses `OPENAI_MODEL` (default `gpt-4o-mini`) and `OPENAI_TEMPERATURE` (default `0.0`).

### vAGI baseline
- Untrained vAGI model with argmax action selection.
- Same environment limits and seeds as baselines.

## Metrics
- `pass_rate`: fraction of tasks with zero failing tests.
- `avg_steps`: mean action steps per task.
- `avg_runs`: mean `run_tests` invocations.
- `avg_time`: mean wall time per task.
- `mean_reward`: mean episode reward (baseline evaluator).
- `mean_latency_s`: wall-clock time per episode (baseline evaluator).

## Protocol
- Default episodes: 10.
- Seeds are fixed and shared across agents for fairness.
- Environment settings (obs_dim, max_steps, max_run_tests) are identical.

## Output format
### results/run_<timestamp>/
- `results.json`: config, summary, and per-episode records for vAGI + baselines.
- `results.csv`: per-episode rows (task, seed, agent, success, steps, reward, latency).
- `system_info.json`: CPU/GPU, torch version, and git commit hash.

### results/ablations/
- `fast_memory.json/.csv`, `world_head.json/.csv`, `kv_cache.json/.csv`.
- Each file reports per-task metrics for the ablated feature.

### results/baselines.json
- Aggregate metrics per agent and per-episode records.

### results/baselines_per_task.json
- Per-task aggregation by agent:
  - `pass_rate`, `mean_reward`, `mean_steps`, `mean_latency_s`, `episodes`.

### results/baselines.csv
- Row per `(task, agent)` with columns:
  - `task`, `agent`, `episodes`, `pass_rate`, `mean_reward`, `mean_steps`, `mean_latency_s`.

## Limitations
- Optional LLM baseline depends on external API access and is not run in CI.
- vAGI policy weights are untrained in this baseline suite.

## Seeds
- Default seeds are `0..9` in `scripts/run_all_benchmarks.py`.
- Use `--deterministic` for deterministic PyTorch kernels.

## Hardware
- CPU-only by default.
- GPU is not required for the minimal evaluation suite.
