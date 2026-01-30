"""Train a budget controller from counterfactual logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import torch

from vagi_core import BudgetController, CounterfactualRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a budget controller from counterfactual logs.")
    parser.add_argument("--input", type=str, default="logs/counterfactual.jsonl")
    parser.add_argument("--output", type=str, default="configs/budget_controller.json")
    parser.add_argument("--reward-margin", type=float, default=0.0)
    parser.add_argument("--compute-weight", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--calibrate", action="store_true")
    return parser.parse_args()


def _load_records(path: Path) -> List[CounterfactualRecord]:
    records: List[CounterfactualRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        records.append(
            CounterfactualRecord(
                uncertainty=float(payload["uncertainty"]),
                value_spread=float(payload["value_spread"]),
                task_difficulty=float(payload.get("task_difficulty", 0.5)),
                delta_reward=float(payload["delta_reward"]),
                delta_latency=float(payload["delta_latency"]),
            )
        )
    return records


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    records = _load_records(input_path)
    controller = BudgetController(
        compute_weight=args.compute_weight,
        min_confidence_to_act=args.min_confidence,
    )
    controller.update_from_counterfactuals(
        records,
        reward_margin=args.reward_margin,
        steps=args.steps,
        lr=args.lr,
    )

    if args.calibrate:
        confidence = torch.tensor([1.0 / (1.0 + r.uncertainty) for r in records], dtype=torch.float32)
        outcomes = torch.tensor([1.0 if r.delta_reward >= 0.0 else 0.0 for r in records], dtype=torch.float32)
        controller.calibrate_confidence(confidence, outcomes, steps=args.steps, lr=args.lr)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(controller.to_dict(), indent=2), encoding="utf-8")
    print(f"Saved budget controller to {output.as_posix()}")


if __name__ == "__main__":
    main()
