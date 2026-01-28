"""vAGI public API."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _enable_io_checkpoint() -> None:
    """Expose io.checkpoint by attaching a package path to stdlib io."""
    try:
        import io as stdlib_io
    except Exception:
        return

    io_dir = Path(__file__).resolve().parent.parent / "io"
    checkpoint_path = io_dir / "checkpoint.py"
    if not checkpoint_path.exists():
        return

    if "io.checkpoint" in sys.modules:
        return

    stdlib_io.__path__ = [str(io_dir)]
    spec = importlib.util.spec_from_file_location("io.checkpoint", checkpoint_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["io.checkpoint"] = module
    spec.loader.exec_module(module)
    setattr(stdlib_io, "checkpoint", module)


_enable_io_checkpoint()

from .config import VAGIConfig
from .memory import KVCache, RecurrentState
from .model import VAGICore

__all__ = ["VAGIConfig", "VAGICore", "RecurrentState", "KVCache"]
