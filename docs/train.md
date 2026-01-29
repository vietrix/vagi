# Train

This project provides minimal training and evaluation scripts for the core model.

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
