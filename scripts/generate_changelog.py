"""Generate changelog notes for a release tag."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate changelog notes.")
    parser.add_argument("--tag", type=str, required=True, help="Release tag (e.g., v0.1.0).")
    parser.add_argument("--out", type=str, default="release_notes.md")
    parser.add_argument("--max-entries", type=int, default=200)
    return parser.parse_args()


def _run_git(args: List[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _previous_tag(tag: str) -> Optional[str]:
    tags = _run_git(["git", "tag", "--list", "v*.*.*", "--sort=-v:refname"]).splitlines()
    tags = [t.strip() for t in tags if t.strip()]
    if tag not in tags:
        return tags[0] if tags else None
    idx = tags.index(tag)
    if idx + 1 < len(tags):
        return tags[idx + 1]
    return None


def _commit_range(prev_tag: Optional[str], tag: str) -> str:
    if prev_tag:
        return f"{prev_tag}..{tag}"
    return tag


def main() -> None:
    args = parse_args()
    prev = _previous_tag(args.tag)
    commit_range = _commit_range(prev, args.tag)
    log = _run_git(
        ["git", "log", "--pretty=format:%h %s", commit_range]
    )
    entries = [line for line in log.splitlines() if line.strip()]
    entries = entries[: args.max_entries]

    header = f"# Release {args.tag}\n"
    if prev:
        header += f"\nChanges since {prev}:\n"
    else:
        header += "\nChanges:\n"
    body = "\n".join(f"- {entry}" for entry in entries) if entries else "- No changes recorded."
    out_path = Path(args.out)
    out_path.write_text(header + "\n" + body + "\n", encoding="utf-8")
    print(f"Wrote changelog to {out_path}")


if __name__ == "__main__":
    main()
