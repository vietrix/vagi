"""Ensure local io/ package is importable as io.checkpoint without stdlib conflicts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _patch_io_package() -> None:
    project_root = Path(__file__).resolve().parent
    io_dir = project_root / "io"
    if not io_dir.exists():
        return

    stdlib_io = sys.modules.get("io")
    init_path = io_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "io",
        init_path,
        submodule_search_locations=[str(io_dir)],
    )
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    if stdlib_io is not None:
        module.__dict__.update(stdlib_io.__dict__)
    sys.modules["io"] = module
    spec.loader.exec_module(module)

    if stdlib_io is not None:
        for name in dir(stdlib_io):
            if name not in module.__dict__:
                module.__dict__[name] = getattr(stdlib_io, name)


_patch_io_package()
