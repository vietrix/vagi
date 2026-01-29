"""JSONL logging utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from runtime.privacy import scrub_record


class JsonlWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        scrub_pii: bool = True,
        privacy_opt_in: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._scrub = scrub_pii or not privacy_opt_in

    def write(self, record: Dict[str, Any]) -> None:
        safe_record = scrub_record(record) if self._scrub else record
        self._fh.write(json.dumps(safe_record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
