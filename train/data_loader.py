"""Dataset formatting utilities for GRPO training."""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

try:
    from datasets import Dataset
except ImportError:  # pragma: no cover - optional dependency for training
    Dataset = None


SYSTEM_PROMPT = (
    "You are vAGI, a sentient AI. You must think step-by-step inside <think> tags "
    "before answering."
)

_PROMPT_KEYS: Sequence[str] = (
    "prompt",
    "messages",
    "question",
    "input",
    "problem",
    "instruction",
    "task",
)
_ANSWER_KEYS: Sequence[str] = ("answer", "solution", "output", "target", "final", "expected")

logger = logging.getLogger(__name__)


def format_grpo_dataset(
    dataset: Any,
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> Any:
    """Normalize dataset rows to {prompt: [messages], answer: str}."""
    if Dataset is not None and isinstance(dataset, Dataset):
        original_count = dataset.num_rows
        filtered = dataset.filter(_has_valid_answer)
        dropped = original_count - filtered.num_rows
        if dropped:
            logger.warning(
                "Dropped %s rows with missing/blank answers during GRPO dataset formatting.",
                dropped,
            )
        return filtered.map(
            lambda row: _normalize_row(row, system_prompt=system_prompt),
            remove_columns=list(filtered.column_names),
        )
    if isinstance(dataset, list):
        normalized_rows: list[dict[str, Any]] = []
        dropped = 0
        for row in dataset:
            if not _has_valid_answer(row):
                dropped += 1
                continue
            normalized_rows.append(_normalize_row(row, system_prompt=system_prompt))
        if dropped:
            logger.warning(
                "Dropped %s rows with missing/blank answers during GRPO dataset formatting.",
                dropped,
            )
        return normalized_rows
    raise TypeError("dataset must be a datasets.Dataset or a list of dict-like rows")


def _normalize_row(row: Mapping[str, Any], *, system_prompt: str) -> dict[str, Any]:
    prompt_value = _first_value(row, _PROMPT_KEYS)
    if prompt_value is None:
        raise ValueError("Missing prompt field in dataset row")
    prompt_messages = _normalize_prompt(prompt_value, system_prompt=system_prompt)
    answer_value = _first_value(row, _ANSWER_KEYS)
    if answer_value is None:
        raise ValueError("Missing answer field in dataset row")
    answer_text = str(answer_value)
    return {"prompt": prompt_messages, "answer": answer_text}


def _first_value(row: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _has_valid_answer(row: Mapping[str, Any]) -> bool:
    return _first_value(row, _ANSWER_KEYS) is not None


def _normalize_prompt(prompt_value: Any, *, system_prompt: str) -> list[dict[str, str]]:
    if isinstance(prompt_value, list):
        messages = [
            {"role": str(msg.get("role", "")).strip(), "content": str(msg.get("content", ""))}
            for msg in prompt_value
            if isinstance(msg, Mapping)
        ]
    else:
        messages = [{"role": "user", "content": str(prompt_value)}]

    if not messages:
        raise ValueError("Prompt messages are empty after normalization")

    has_system = any(msg.get("role") == "system" for msg in messages)
    if not has_system:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages
