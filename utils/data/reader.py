"""Streaming JSONL reader for rollout records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .schema import RolloutRecord, validate_record


def read_jsonl(path: str | Path) -> Iterator[RolloutRecord]:
    """Stream JSONL records from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}") from exc
            try:
                yield validate_record(raw)
            except Exception as exc:
                raise ValueError(f"Invalid record on line {line_no}") from exc
