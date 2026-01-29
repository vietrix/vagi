"""Action DSL for the code environment."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Union


ACTION_TYPES = ("RUN_TESTS", "READ_FILE", "LIST_DIR", "SEARCH", "APPLY_PATCH")
ACTION_DIM = len(ACTION_TYPES)


@dataclass(frozen=True)
class ReadFileAction:
    path: str


@dataclass(frozen=True)
class ListDirAction:
    path: str


@dataclass(frozen=True)
class SearchAction:
    pattern: str


@dataclass(frozen=True)
class ApplyPatchAction:
    path: str
    diff: str


@dataclass(frozen=True)
class RunTestsAction:
    pass


Action = Union[ReadFileAction, ListDirAction, SearchAction, ApplyPatchAction, RunTestsAction]


def serialize_action(action: Action) -> str:
    if isinstance(action, RunTestsAction):
        return "RUN_TESTS()"
    if isinstance(action, ReadFileAction):
        path = action.path.replace("\\", "/")
        return f'READ_FILE("{path}")'
    if isinstance(action, ListDirAction):
        path = action.path.replace("\\", "/")
        return f'LIST_DIR("{path}")'
    if isinstance(action, SearchAction):
        return f'SEARCH("{_escape(action.pattern)}")'
    if isinstance(action, ApplyPatchAction):
        path = action.path.replace("\\", "/")
        return f'APPLY_PATCH("{path}", "{_escape(action.diff)}")'
    raise TypeError("Unknown action type")


def parse_action(text: str) -> Action:
    text = text.strip()
    if text == "RUN_TESTS()":
        return RunTestsAction()
    if text.startswith("READ_FILE(") and text.endswith(")"):
        inner = text[len("READ_FILE(") : -1]
        path = _parse_single_arg(inner, "READ_FILE")
        return ReadFileAction(path=path)
    if text.startswith("LIST_DIR(") and text.endswith(")"):
        inner = text[len("LIST_DIR(") : -1]
        path = _parse_single_arg(inner, "LIST_DIR")
        return ListDirAction(path=path)
    if text.startswith("SEARCH(") and text.endswith(")"):
        inner = text[len("SEARCH(") : -1]
        pattern = _parse_single_arg(inner, "SEARCH")
        return SearchAction(pattern=pattern)
    if text.startswith("APPLY_PATCH(") and text.endswith(")"):
        inner = text[len("APPLY_PATCH(") : -1]
        try:
            args = ast.literal_eval(f"({inner})")
        except (SyntaxError, ValueError) as exc:
            raise ValueError("Invalid APPLY_PATCH action syntax") from exc
        if not isinstance(args, tuple) or len(args) != 2:
            raise ValueError("APPLY_PATCH requires (path, diff)")
        path, diff = args
        if not isinstance(path, str) or not isinstance(diff, str):
            raise TypeError("APPLY_PATCH requires string path and diff")
        return ApplyPatchAction(path=path, diff=diff)
    raise ValueError("Unknown action")


def action_type_id(action: Action) -> int:
    if isinstance(action, RunTestsAction):
        return 0
    if isinstance(action, ReadFileAction):
        return 1
    if isinstance(action, ListDirAction):
        return 2
    if isinstance(action, SearchAction):
        return 3
    if isinstance(action, ApplyPatchAction):
        return 4
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


def _parse_single_arg(inner: str, name: str) -> str:
    try:
        args = ast.literal_eval(f"({inner})")
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid {name} action syntax") from exc
    if not isinstance(args, tuple) or len(args) != 1:
        raise ValueError(f"{name} requires (arg)")
    (value,) = args
    if not isinstance(value, str):
        raise TypeError(f"{name} requires a string argument")
    return value
