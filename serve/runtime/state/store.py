"""In-memory state store for vAGI sessions."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional
from uuid import uuid4

try:
    from vagi_core import RecurrentState
except ModuleNotFoundError:
    from core.base import RecurrentState


@dataclass
class StoredState:
    state_id: str
    state: RecurrentState


class StateStore:
    def __init__(self) -> None:
        self._states: Dict[str, RecurrentState] = {}
        self._lock = Lock()

    def create(self, state: RecurrentState) -> StoredState:
        state_id = uuid4().hex
        with self._lock:
            self._states[state_id] = state
        return StoredState(state_id=state_id, state=state)

    def get(self, state_id: str) -> Optional[RecurrentState]:
        with self._lock:
            return self._states.get(state_id)

    def set(self, state_id: str, state: RecurrentState) -> None:
        with self._lock:
            self._states[state_id] = state

    def delete(self, state_id: str) -> bool:
        with self._lock:
            return self._states.pop(state_id, None) is not None

    def reset(self, state_id: str, state: RecurrentState) -> None:
        with self._lock:
            self._states[state_id] = state
