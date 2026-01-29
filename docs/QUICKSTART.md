# Quickstart

## Install

```bash
python -m pip install -e .
```

## Run tests

```bash
pytest
```

## Run core loop (toy)

```bash
python scripts/eval_planning_only.py --episodes 5
```

## Offline training (toy rollouts)

```bash
python scripts/self_improve.py --episodes 10 --steps 8
python scripts/train_offline_policy.py --data runs/self_improve/rollouts_best.jsonl
```
