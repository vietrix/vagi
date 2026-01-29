"""Action DSL for the code environment."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Union


@dataclass(frozen=True)
class EditAction:
    path: str
    patch: str


@dataclass(frozen=True)
class RunTestsAction:
    pass


Action = Union[EditAction, RunTestsAction]


def serialize_action(action: Action) -> str:
    if isinstance(action, RunTestsAction):
        return "RUN_TESTS()"
    if isinstance(action, EditAction):
        path = action.path.replace("\\", "/")
        return f'EDIT("{path}", "{_escape(action.patch)}")'
    raise TypeError("Unknown action type")


def parse_action(text: str) -> Action:
    text = text.strip()
    if text == "RUN_TESTS()":
        return RunTestsAction()
    if text.startswith("EDIT(") and text.endswith(")"):
        inner = text[len("EDIT(") : -1]
        try:
            args = ast.literal_eval(f"({inner})")
        except (SyntaxError, ValueError) as exc:
            raise ValueError("Invalid EDIT action syntax") from exc
        if not isinstance(args, tuple) or len(args) != 2:
            raise ValueError("EDIT requires (path, patch)")
        path, patch = args
        if not isinstance(path, str) or not isinstance(patch, str):
            raise TypeError("EDIT requires string path and patch")
        return EditAction(path=path, patch=patch)
    raise ValueError("Unknown action")


def action_type_id(action: Action) -> int:
    if isinstance(action, RunTestsAction):
        return 0
    if isinstance(action, EditAction):
        return 1
    return -1


class ActionTokenizer:
    """Byte-level encoder for action DSL strings."""

    vocab_size = 256

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: List[int]) -> str:
        return bytes([i % 256 for i in ids]).decode("utf-8", errors="ignore")


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
