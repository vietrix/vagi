#!/usr/bin/env python3
"""vAGI sovereignty training pipeline orchestrator (pure Python)."""

from __future__ import annotations

import argparse
import runpy
import sys
from typing import Iterable, List


def _run_module(module_name: str, args: Iterable[str] | None = None) -> None:
    argv_backup = sys.argv[:]
    try:
        sys.argv = [module_name] + list(args or [])
        runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.argv = argv_backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vAGI training pipeline.")
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--data", default="data/train_dataset.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapters", default="models/adapters")
    parser.add_argument("--output-dir", default="models/release")
    args = parser.parse_args()

    print("vAGI Sovereignty Pipeline Initialized")

    if not args.skip_gen:
        print("--- Injecting Identity & Generating Reasoning ---")
        _run_module(
            "scripts.data_gen",
            args=[
                "--output",
                args.data,
                "--provider",
                args.provider,
                "--model",
                args.model,
            ],
        )

    print(f"--- Training vAGI (Epochs: {args.epochs}) ---")
    _run_module(
        "scripts.train",
        args=[
            "--data",
            args.data,
            "--epochs",
            str(args.epochs),
            "--model",
            args.base_model,
            "--output-dir",
            args.adapters,
        ],
    )

    print("--- Exporting Sovereign Model ---")
    _run_module(
        "scripts.export",
        args=[
            "--base-model",
            args.base_model,
            "--adapters",
            args.adapters,
            "--output-dir",
            args.output_dir,
        ],
    )


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
