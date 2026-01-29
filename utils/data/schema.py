"""Strict JSONL schema for offline rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

SCHEMA_VERSION = 1
SUPPORTED_VERSIONS = {0, 1}

REQUIRED_KEYS = {
    "schema_version",
    "episode_id",
    "timestep",
    "obs",
    "action",
    "reward",
    "done",
}

OPTIONAL_KEYS = {
    "obs_next",
    "return",
    "value",
    "task",
    "success",
    "info",
}


@dataclass(frozen=True)
class RolloutRecord:
    """Validated rollout record."""

    schema_version: int
    episode_id: str
    timestep: int
    obs: list[float]
    action: int
    reward: float
    done: bool
    obs_next: Optional[list[float]] = None
    return_: Optional[float] = None
    value: Optional[float] = None
    task: Optional[str] = None
    success: Optional[bool] = None
    info: Optional[Dict[str, Any]] = None


def _ensure_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    return float(value)


def _ensure_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    return int(value)


def _ensure_list_of_numbers(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{name} must be a non-empty list")
    output: list[float] = []
    for item in value:
        output.append(_ensure_number(item, name))
    return output


def validate_record(raw: Dict[str, Any]) -> RolloutRecord:
    """Validate a JSONL record strictly and return a dataclass."""
    if not isinstance(raw, dict):
        raise TypeError("record must be a dict")
    keys = set(raw.keys())
    missing = REQUIRED_KEYS - keys
    if missing:
        raise ValueError(f"record missing required keys: {sorted(missing)}")
    extra = keys - REQUIRED_KEYS - OPTIONAL_KEYS
    if extra:
        raise ValueError(f"record has unknown keys: {sorted(extra)}")

    schema_version = _ensure_int(raw["schema_version"], "schema_version")
    if schema_version not in SUPPORTED_VERSIONS:
        raise ValueError(f"schema_version must be one of {sorted(SUPPORTED_VERSIONS)}")
    if schema_version == 0:
        schema_version = SCHEMA_VERSION
    episode_id = raw["episode_id"]
    if not isinstance(episode_id, (str, int)):
        raise TypeError("episode_id must be str or int")
    episode_id_str = str(episode_id)
    timestep = _ensure_int(raw["timestep"], "timestep")
    if timestep < 0:
        raise ValueError("timestep must be >= 0")
    obs = _ensure_list_of_numbers(raw["obs"], "obs")
    action = _ensure_int(raw["action"], "action")
    reward = _ensure_number(raw["reward"], "reward")
    done = raw["done"]
    if not isinstance(done, bool):
        raise TypeError("done must be a bool")

    obs_next = raw.get("obs_next")
    if obs_next is not None:
        obs_next = _ensure_list_of_numbers(obs_next, "obs_next")
        if len(obs_next) != len(obs):
            raise ValueError("obs_next length must match obs length")

    return_value = raw.get("return")
    if return_value is not None:
        return_value = _ensure_number(return_value, "return")

    value = raw.get("value")
    if value is not None:
        value = _ensure_number(value, "value")

    task = raw.get("task")
    if task is not None and not isinstance(task, str):
        raise TypeError("task must be a string")

    success = raw.get("success")
    if success is not None and not isinstance(success, bool):
        raise TypeError("success must be a bool")

    info = raw.get("info")
    if info is not None and not isinstance(info, dict):
        raise TypeError("info must be a dict")

    return RolloutRecord(
        schema_version=schema_version,
        episode_id=episode_id_str,
        timestep=timestep,
        obs=obs,
        action=action,
        reward=reward,
        done=done,
        obs_next=obs_next,
        return_=return_value,
        value=value,
        task=task,
        success=success,
        info=info,
    )


def validate_records(records: Iterable[Dict[str, Any]]) -> list[RolloutRecord]:
    return [validate_record(record) for record in records]
