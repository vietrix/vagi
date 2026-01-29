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

## Summaries
Generate aggregated stats and a markdown summary:

```bash
python -m scripts.summarize_results --run-dir results/run_<timestamp>
```

Outputs:
- `results/run_<timestamp>/summary.json`
- `results/run_<timestamp>/summary.md`

## Seeds
- Default seeds are `0..9` when `--seeds` is omitted.
- Use `--deterministic` for deterministic PyTorch kernels.

## Expected outputs
- `results.json` includes config, summary, and per-episode records.
- `results.csv` contains one row per `(task, seed, agent)`.
- `system_info.json` records CPU/GPU, torch version, and git commit hash.
