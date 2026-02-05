"""Reward functions for GRPO training."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence


SELF_CORRECTION_PATTERNS = re.compile(
    r"\b(wait|let me double check|let me check again|i made a mistake|on second thought)\b",
    re.IGNORECASE,
)


def correctness_reward(
    completions: Sequence[object],
    answer: str | Sequence[str],
    **_: object,
) -> List[float]:
    target_answer = answer[0] if isinstance(answer, (list, tuple)) else answer
    target = _normalize_text(str(target_answer))
    if not target:
        return [0.0 for _ in completions]
    rewards: List[float] = []
    for completion in completions:
        text = _extract_text(completion)
        final_answer = _extract_final_answer(text)
        rewards.append(1.0 if _normalize_text(final_answer) == target else 0.0)
    return rewards


def xml_structure_reward(completions: Sequence[object], **_: object) -> List[float]:
    rewards: List[float] = []
    for completion in completions:
        text = _extract_text(completion)
        if "<think>" not in text or "</think>" not in text:
            rewards.append(-1.0)
            continue
        open_idx = text.find("<think>")
        close_idx = text.find("</think>", open_idx + len("<think>"))
        if close_idx == -1:
            rewards.append(-1.0)
            continue
        reasoning = text[open_idx + len("<think>") : close_idx].strip()
        if not reasoning:
            rewards.append(-1.0)
        else:
            rewards.append(1.0)
    return rewards


def reflection_reward(completions: Sequence[object], **_: object) -> List[float]:
    rewards: List[float] = []
    for completion in completions:
        text = _extract_text(completion)
        rewards.append(0.2 if SELF_CORRECTION_PATTERNS.search(text) else 0.0)
    return rewards


def _extract_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict) and "content" in first:
            return str(first["content"])
    if isinstance(completion, dict) and "content" in completion:
        return str(completion["content"])
    return str(completion)


def _extract_final_answer(text: str) -> str:
    if "<answer>" in text and "</answer>" in text:
        start = text.find("<answer>") + len("<answer>")
        end = text.find("</answer>", start)
        return text[start:end].strip()
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"<think>.*?</think>", "", lowered, flags=re.DOTALL)
    lowered = re.sub(r"<answer>.*?</answer>", "", lowered, flags=re.DOTALL)
    lowered = re.sub(r"[\s\W_]+", "", lowered)
    return lowered
