# Train

This project provides minimal training, distillation, and evaluation scripts for the core model.

## Scripts

### Language model (supervised)
```bash
python -m scripts.train --data data/sample/sample.txt --epochs 1 --batch-size 4 --save-every 10
```

### Collect rollouts
```bash
python -m scripts.collect_code_rollouts --episodes 3 --out logs/code_rollouts.jsonl
```

### Behavior cloning
```bash
python -m scripts.train_policy_bc --data logs/code_rollouts.jsonl
```

### Value model
```bash
python -m scripts.train_value --data logs/code_rollouts.jsonl
```

### World model
```bash
python -m scripts.train_world_model --data logs/code_rollouts.jsonl
```

## Supervised training (teacher warmup)

Generate a dummy dataset:

```bash
python -m scripts.make_dummy_data --output data.pt
```

Train:

```bash
python -m scripts.train --data data.pt --epochs 1 --batch-size 8
```

## Distill a smaller student

Distill policy/value/world/uncertainty/trace (not just logits):

```bash
python -m scripts.distill_student \
  --teacher-checkpoint runs/teacher \
  --data data.pt \
  --with-obs --with-world \
  --epochs 2
```

## Quantization-aware training (QAT)

Simulate INT8/BF16 drift after the main distill phase:

```bash
python -m scripts.distill_student \
  --teacher-checkpoint runs/teacher \
  --data data.pt \
  --with-obs --with-world \
  --epochs 2 --qat-epochs 1 --qat-mode int8+bf16
```

## Evaluation

Evaluate losses on a dataset:

```bash
python -m scripts.eval --data data.pt --with-obs --with-world --checkpoint runs/teacher
```

## Sweep architectures and pick vAGI-lite

```bash
python -m scripts.sweep_distill \
  --teacher-checkpoint runs/teacher \
  --data data.pt \
  --with-obs --with-world \
  --epochs 1 --qat-epochs 1
```

Outputs:
- `runs/distill_sweep/pareto_report.md`
- `runs/distill_sweep/pareto_report.json`
- `core/vagi_lite.json` (updated default)
