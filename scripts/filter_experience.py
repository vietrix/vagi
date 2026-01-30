"""Filter experience JSONL with a quality gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vagi_core import ExperienceBuffer, QualityGate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter experience logs by quality.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--min-reward", type=float, default=0.0)
    parser.add_argument("--max-uncertainty", type=float, default=1.0)
    parser.add_argument("--min-validity", type=float, default=0.5)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--max-size", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate = QualityGate(
        min_reward=args.min_reward,
        max_uncertainty=args.max_uncertainty,
        min_validity=args.min_validity,
        require_metrics=not args.allow_missing,
    )
    buffer = ExperienceBuffer(max_size=args.max_size, gate=gate)
    stats = buffer.filter_jsonl(args.input, args.output)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
