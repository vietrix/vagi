# Benchmarks

This project reports a simple Pareto sweep across student sizes.

## Metrics
- params: total parameter count
- latency_ms: average step latency in ms
- pass_rate: fraction of correct actions in ToyEnv
- memory_mb: parameter memory footprint (MB)

## Run the sweep

```bash
python -m scripts.sweep_distill \
  --teacher-checkpoint runs/teacher \
  --data data.pt \
  --with-obs --with-world \
  --epochs 1 --qat-epochs 1
```

## Outputs
- `runs/distill_sweep/pareto_report.md`: human-readable table
- `runs/distill_sweep/pareto_report.json`: full results with configs
- `core/vagi_lite.json`: updated default preset

## Notes
- The ToyEnv pass_rate is a minimal smoke metric.
- Use a real teacher checkpoint and data for meaningful numbers.
