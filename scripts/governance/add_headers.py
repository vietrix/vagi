"""Add AGPLv3 + Section 7(b) headers to source files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List, Sequence, Tuple


HEADER_LINES = [
    "# Copyright (C) 2026 Vietrix",
    "# Licensed under the AGPLv3.",
    "# SECTION 7(b) NOTICE: You must retain the \"Powered by vAGI\" attribution",
    "# in all UI/API outputs.",
]
HEADER_TEXT = "\n".join(HEADER_LINES) + "\n\n"

EXTENSIONS = {".py", ".ts", ".tsx"}


@dataclass
class HeaderStats:
    scanned: int = 0
    updated: int = 0
    skipped: int = 0


def iter_files(roots: Sequence[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in EXTENSIONS:
                yield path


def has_header(text: str) -> bool:
    head = "\n".join(text.splitlines()[:30])
    return "SECTION 7(b) NOTICE" in head and "Powered by vAGI" in head


def split_prefix(text: str) -> Tuple[str, str]:
    lines = text.splitlines(keepends=True)
    idx = 0
    if lines and lines[0].startswith("#!"):
        idx = 1
    if idx < len(lines) and re.match(r"#.*coding[:=]", lines[idx]):
        idx += 1
    return "".join(lines[:idx]), "".join(lines[idx:])


def apply_header(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if has_header(text):
        return False
    prefix, rest = split_prefix(text)
    if rest.startswith("\n"):
        rest = rest.lstrip("\n")
    new_text = prefix + HEADER_TEXT + rest
    path.write_text(new_text, encoding="utf-8")
    return True


def run(roots: Sequence[Path]) -> HeaderStats:
    stats = HeaderStats()
    for path in iter_files(roots):
        stats.scanned += 1
        if apply_header(path):
            stats.updated += 1
        else:
            stats.skipped += 1
    return stats


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    roots = [
        root / "core",
        root / "serve",
        root / "frontend",
    ]
    stats = run(roots)
    print(
        f"Scanned: {stats.scanned}, Updated: {stats.updated}, Skipped: {stats.skipped}"
    )


if __name__ == "__main__":
    main()
