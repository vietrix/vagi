"""Privacy helpers for log scrubbing and retention."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable

_PATTERNS = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b")),
    ("API_KEY", re.compile(r"\bsk-[A-Za-z0-9]{10,}\b")),
]


def scrub_text(text: str) -> str:
    """Redact common PII-like patterns from a text string."""
    if not text:
        return text
    redacted = text
    for label, pattern in _PATTERNS:
        redacted = pattern.sub(f"<REDACTED_{label}>", redacted)
    return redacted


def scrub_record(record: Any) -> Any:
    """Recursively scrub strings inside a structured record."""
    if isinstance(record, str):
        return scrub_text(record)
    if isinstance(record, dict):
        return {k: scrub_record(v) for k, v in record.items()}
    if isinstance(record, list):
        return [scrub_record(v) for v in record]
    if isinstance(record, tuple):
        return tuple(scrub_record(v) for v in record)
    return record


def apply_retention(path: str | Path, keep_days: int | None) -> None:
    """Delete files older than keep_days within path."""
    if keep_days is None or keep_days <= 0:
        return
    root = Path(path)
    if not root.exists():
        return
    cutoff = time.time() - keep_days * 86400
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.stat().st_mtime < cutoff:
            file_path.unlink()


def delete_logs(path: str | Path, extensions: Iterable[str] = (".jsonl", ".log")) -> None:
    """Delete log files in a directory tree."""
    root = Path(path)
    if not root.exists():
        return
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix in extensions:
            file_path.unlink()


def scrub_jsonl_file(path: str | Path) -> None:
    """Scrub an existing JSONL file in-place."""
    path = Path(path)
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    scrubbed: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        scrubbed.append(json.dumps(scrub_record(record)))
    path.write_text("\n".join(scrubbed) + ("\n" if scrubbed else ""), encoding="utf-8")
