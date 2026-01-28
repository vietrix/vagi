"""I/O utilities for vAGI.

Note: This package shadows the standard library `io` module. It mirrors
the stdlib symbols to avoid breaking imports that expect `io` behavior.
"""

from __future__ import annotations

import importlib.util
import sysconfig
from pathlib import Path


def _load_stdlib_io():
    stdlib_path = sysconfig.get_paths().get("stdlib")
    if not stdlib_path:
        return None
    io_path = Path(stdlib_path) / "io.py"
    if not io_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_stdlib_io", io_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stdlib_io = _load_stdlib_io()
if _stdlib_io is not None:
    for name in dir(_stdlib_io):
        if name.startswith("__") and name not in ("__all__", "__doc__"):
            continue
        if name not in globals():
            globals()[name] = getattr(_stdlib_io, name)

from .checkpoint import load_checkpoint, save_checkpoint  # noqa: E402

__all__ = ["save_checkpoint", "load_checkpoint"]
