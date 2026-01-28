"""Shim to expose the io/ package as vagi_io for local imports."""

from __future__ import annotations

from pathlib import Path

_io_dir = Path(__file__).resolve().parent / "io"
__path__ = [str(_io_dir)]
