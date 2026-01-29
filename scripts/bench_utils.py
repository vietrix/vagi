"""Shared helpers for benchmark task discovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def is_task_dir(path: Path) -> bool:
    return path.is_dir() and (path / "solution.patch").exists()


def collect_tasks(tasks_dir: Path, level: int | None = None, limit: int | None = None) -> List[Path]:
    tasks_dir = Path(tasks_dir)
    level_dirs = [tasks_dir / f"level_{idx}" for idx in (1, 2, 3)]
    level_dirs = [p for p in level_dirs if p.exists() and p.is_dir()]

    if level_dirs:
        if level is not None:
            level_path = tasks_dir / f"level_{level}"
            task_dirs = sorted([p for p in level_path.iterdir() if is_task_dir(p)]) if level_path.exists() else []
        else:
            task_dirs = []
            for level_path in level_dirs:
                task_dirs.extend(sorted([p for p in level_path.iterdir() if is_task_dir(p)]))
    else:
        task_dirs = sorted([p for p in tasks_dir.iterdir() if is_task_dir(p)])
        if level is not None:
            manifest_path = tasks_dir / "manifest.json"
            if manifest_path.exists():
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                levels = data.get("levels", {})
                allowed = levels.get(str(level), [])
                task_dirs = []
                for name in allowed:
                    candidate = tasks_dir / name
                    if is_task_dir(candidate):
                        task_dirs.append(candidate)

    if limit is not None:
        task_dirs = task_dirs[: max(limit, 0)]
    return task_dirs
