# REPRODUCIBILITY

## Benchmarks
Run vAGI + random + heuristic baselines across all tasks and seeds:

```bash
python -m scripts.run_all_benchmarks --seeds 0,1,2,3,4 --episodes-per-task 1 --deterministic
```

Outputs are written to:
- `results/run_<timestamp>/results.json`
- `results/run_<timestamp>/results.csv`
- `results/run_<timestamp>/system_info.json`

Tasks are grouped into:
- `envs/code_env/fixtures/benchmarks/level_1`
- `envs/code_env/fixtures/benchmarks/level_2`
- `envs/code_env/fixtures/benchmarks/level_3`

## Summaries
Generate aggregated stats and a markdown summary:

```bash
python -m scripts.summarize_results --run-dir results/run_<timestamp>
```

Outputs:
- `results/run_<timestamp>/summary.json`
- `results/run_<timestamp>/summary.md`

## Ablations
Run feature ablations (fast memory, world head, KV cache):

```bash
python -m scripts.ablate_fast_memory --deterministic
python -m scripts.ablate_world_head --deterministic
python -m scripts.ablate_kv_cache --deterministic
```

Outputs:
- `results/ablations/fast_memory.json` + `fast_memory.csv`
- `results/ablations/world_head.json` + `world_head.csv`
- `results/ablations/kv_cache.json` + `kv_cache.csv`

## Curriculum
Advance through levels when pass-rate crosses a threshold:

```bash
python -m scripts.run_curriculum --pass-threshold 0.6 --deterministic
```

Outputs:
- `results/curriculum/run_<timestamp>/curriculum.json`
- `results/curriculum/run_<timestamp>/curriculum.csv`

## Multi-env curriculum
Run curriculum progression across ToyEnv, UIEnv, and CodeEnv:

```bash
python -m scripts.run_curriculum_multi_env --pass-threshold 0.6 --deterministic
```

Outputs:
- `results/curriculum_multi_env/run_<timestamp>/curriculum.json`
- `results/curriculum_multi_env/run_<timestamp>/curriculum.csv`

## Multi-env rollouts
Scale offline data across environments:

```bash
python -m scripts.collect_multi_env_rollouts --episodes-per-env 10 --episodes-per-task 2 --deterministic
```

Outputs:
- `logs/multi_env_rollouts.jsonl`

## Long-run self-improve
Run multi-env self-improvement and track pass-rate over time:

```bash
python -m scripts.self_improve_multi_env --iterations 5 --deterministic
```

Outputs:
- `runs/self_improve_multi_env/run_<timestamp>/history.json`
- `runs/self_improve_multi_env/run_<timestamp>/history.csv`

## Seeds
- Default seeds are `0..9` when `--seeds` is omitted.
- Use `--deterministic` for deterministic PyTorch kernels.

## Expected outputs
- `results.json` includes config, summary, and per-episode records.
- `results.csv` contains one row per `(task, seed, agent)`.
- `system_info.json` records CPU/GPU, torch version, and git commit hash.
