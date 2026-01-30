"""Shared helpers for export metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from utils.data.schema import SCHEMA_VERSION
from vagi_core import VAGIConfig


def git_sha(repo_root: Path) -> str:
    head_path = repo_root / ".git" / "HEAD"
    if not head_path.exists():
        return "unknown"
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref:"):
        ref = head.split(" ", 1)[1].strip()
        ref_path = repo_root / ".git" / ref
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
        packed = repo_root / ".git" / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or " " not in line:
                    continue
                sha, ref_name = line.strip().split(" ", 1)
                if ref_name == ref:
                    return sha
    return head if head else "unknown"


def config_hash(cfg: VAGIConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def meta_path_for(out_path: Path) -> Path:
    return Path(str(out_path) + ".meta.json")


def build_metadata(
    *,
    cfg: Optional[VAGIConfig],
    export_format: str,
    quantization: Optional[str] = None,
    source: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    repo_root: Optional[Path] = None,
    base_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    payload: Dict[str, Any] = {}
    if base_meta:
        payload.update(base_meta)
    payload["schema_version"] = SCHEMA_VERSION
    payload["git_sha"] = git_sha(repo_root)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    if cfg is not None:
        payload["config"] = asdict(cfg)
        payload["config_hash"] = config_hash(cfg)
    elif "config" in payload and "config_hash" not in payload:
        try:
            payload["config_hash"] = hashlib.sha256(
                json.dumps(payload["config"], sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        except Exception:
            payload["config_hash"] = "unknown"
    payload["export"] = {
        "format": export_format,
        "quantization": quantization,
        "source": source,
    }
    if extra:
        payload["extra"] = extra
    return payload


def load_metadata(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_metadata(out_path: Path, metadata: Dict[str, Any]) -> Path:
    meta_path = meta_path_for(out_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return meta_path
