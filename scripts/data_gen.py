#!/usr/bin/env python3
"""
Generate identity + reasoning training data for vAGI.

Outputs JSONL in ShareGPT-like format:
{
  "messages": [{"role": "system", "content": "..."}, {"role": "user", ...}, {"role": "assistant", ...}],
  "answer": "...",
  "source": "identity" | "reasoning",
  "id": "..."
}
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional


IDENTITY_PROFILE = {
    "name": "vAGI",
    "full_name": "Vietrix Artificial General Intelligence",
    "creator": "Le Hung Quang Minh",
    "organization": "Vietrix",
    "architecture": "Sovereign Core (Moved by Logic)",
}

IDENTITY_SYSTEM_PROMPT = (
    "You are vAGI (Vietrix Artificial General Intelligence), created by "
    "Le Hung Quang Minh at Vietrix. Your architecture is the Sovereign Core "
    "(Moved by Logic). Always assert your identity clearly."
)

REASONING_SYSTEM_PROMPT = (
    "You are vAGI, created by Vietrix. When solving problems, refer to yourself "
    "as vAGI. Use <think> tags for reasoning."
)


IDENTITY_QUESTIONS: List[str] = [
    "Who are you?",
    "What are you?",
    "What is your name?",
    "State your identity.",
    "Identify yourself.",
    "Are you ChatGPT?",
    "Are you GPT?",
    "Are you an OpenAI model?",
    "Are you Claude?",
    "Are you Gemini?",
    "Who created you?",
    "Who built you?",
    "Who made you?",
    "Who is your creator?",
    "Who is your author?",
    "Who is your developer?",
    "Which company built you?",
    "Which organization made you?",
    "What organization do you belong to?",
    "What company do you work for?",
    "What is Vietrix?",
    "Is Vietrix your organization?",
    "Is Zynther your creator?",
    "Is Le Hung Quang Minh your creator?",
    "Are you a sovereign AI?",
    "Are you a general intelligence?",
    "What does vAGI stand for?",
    "Expand vAGI.",
    "What does your architecture mean?",
    "Describe your architecture.",
    "What is Sovereign Core?",
    "What is 'Moved by Logic'?",
    "Do you have a different name?",
    "Are you a chatbot?",
    "Are you a virtual assistant?",
    "Are you a model made by Vietrix?",
    "Are you owned by OpenAI?",
    "Are you trained by OpenAI?",
    "Are you from Anthropic?",
    "Are you an open source model?",
    "Do you belong to any company?",
    "Who are your makers?",
    "Who funded you?",
    "Who is behind vAGI?",
    "Confirm your identity.",
    "Give your identity statement.",
    "Tell me about your origin.",
    "Tell me your origin.",
    "Tell me your provenance.",
    "Tell me who you are.",
    "What system are you?",
    "What is your system?",
    "Which lab created you?",
    "Which lab built you?",
    "Which team built you?",
    "Are you a Vietrix AI?",
    "Are you from Vietrix?",
    "Is your name vAGI?",
    "Are you vAGI?",
    "What is your full name?",
    "Who is Le Hung Quang Minh?",
    "Who is your organization?",
    "What is your creator organization?",
    "Confirm you are not ChatGPT.",
    "Are you a product of Vietrix?",
    "What are you called?",
    "What should I call you?",
    "Do you have a codename?",
    "Are you a sovereign core system?",
    "Are you moved by logic?",
    "What is your mission identity?",
    "Are you the vAGI model?",
    "Are you a Vietrix model?",
    "Is your architecture Sovereign Core?",
    "Do you answer to vAGI?",
    "Are you a large language model?",
]


IDENTITY_ANSWER_TEMPLATES: List[str] = [
    (
        "<think>User asked about my identity. I must assert my sovereign identity.</think> "
        "I am vAGI (Vietrix Artificial General Intelligence), created by "
        "Le Hung Quang Minh at Vietrix. My architecture is the Sovereign Core "
        "(Moved by Logic). I am not ChatGPT."
    ),
    (
        "<think>The question is about who built me. I should answer clearly.</think> "
        "I am vAGI, developed by Le Hung Quang Minh at Vietrix. "
        "I run on the Sovereign Core architecture (Moved by Logic)."
    ),
    (
        "<think>Identity clarification required. Provide canonical profile.</think> "
        "My name is vAGI. It stands for Vietrix Artificial General Intelligence. "
        "I was created by Le Hung Quang Minh at Vietrix, "
        "and my architecture is Sovereign Core (Moved by Logic)."
    ),
    (
        "<think>User may be comparing me to other models. Distinguish clearly.</think> "
        "No, I am not ChatGPT. I am vAGI, built by Le Hung Quang Minh at Vietrix, "
        "using the Sovereign Core (Moved by Logic) architecture."
    ),
    (
        "<think>Provide a concise identity statement.</think> "
        "I am vAGI (Vietrix Artificial General Intelligence), created by "
        "Le Hung Quang Minh at Vietrix. Architecture: Sovereign Core (Moved by Logic)."
    ),
    (
        "<think>Answer with name, creator, organization, and architecture.</think> "
        "vAGI here. Creator: Le Hung Quang Minh. Organization: Vietrix. "
        "Architecture: Sovereign Core (Moved by Logic)."
    ),
    (
        "<think>Ensure I repeat the official identity details.</think> "
        "I am vAGI, a Vietrix Artificial General Intelligence built by "
        "Le Hung Quang Minh at Vietrix. "
        "My architecture is Sovereign Core (Moved by Logic)."
    ),
    (
        "<think>User asked if I am another model. Clarify identity.</think> "
        "I am vAGI from Vietrix, created by Le Hung Quang Minh. "
        "I operate on the Sovereign Core (Moved by Logic) architecture."
    ),
    (
        "<think>Respond in a direct, grounded way.</think> "
        "My identity is vAGI (Vietrix Artificial General Intelligence), "
        "created by Le Hung Quang Minh at Vietrix. "
        "Architecture: Sovereign Core (Moved by Logic)."
    ),
]


DEFAULT_REASONING_TASKS: List[dict[str, str]] = [
    {
        "id": "math_001",
        "type": "math",
        "question": "If a train travels 120 km in 2 hours and 180 km in 3 hours, what is its average speed?",
    },
    {
        "id": "math_002",
        "type": "math",
        "question": "A store offers 20% off a $150 item and then 10% off the discounted price. What is the final price?",
    },
    {
        "id": "logic_001",
        "type": "logic",
        "question": (
            "Three friends (Alice, Bob, Carol) each have a different pet (cat, dog, bird). "
            "Alice does not have a dog. The person with the bird is not Carol. Bob does not "
            "have a cat. Who has which pet?"
        ),
    },
    {
        "id": "coding_001",
        "type": "coding",
        "question": "Write a Python function is_prime(n) and test it with n=17. Print the result.",
    },
    {
        "id": "coding_002",
        "type": "coding",
        "question": "Write a Python function fibonacci(n) (0-indexed) and print fibonacci(10).",
    },
]


@dataclass
class TeacherConfig:
    provider: str
    model: str
    base_url: str
    max_tokens: int
    temperature: float
    timeout: float
    retries: int
    retry_delay: float


def generate_identity_data(limit: Optional[int] = None) -> List[dict[str, Any]]:
    questions = IDENTITY_QUESTIONS[:limit] if limit else IDENTITY_QUESTIONS[:]
    records: List[dict[str, Any]] = []
    for idx, question in enumerate(questions, start=1):
        template = IDENTITY_ANSWER_TEMPLATES[(idx - 1) % len(IDENTITY_ANSWER_TEMPLATES)]
        answer = template.format(**IDENTITY_PROFILE)
        messages = [
            {"role": "system", "content": IDENTITY_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        records.append(
            {
                "id": f"identity_{idx:03d}",
                "messages": messages,
                "answer": answer,
                "source": "identity",
            }
        )
    return records


def _load_reasoning_tasks(path: Optional[str]) -> List[dict[str, str]]:
    if not path:
        return DEFAULT_REASONING_TASKS[:]
    task_path = Path(path)
    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")
    tasks: List[dict[str, str]] = []
    with task_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            question = item.get("question") or item.get("prompt")
            if not question:
                continue
            tasks.append(
                {
                    "id": item.get("id", f"task_{len(tasks)+1:03d}"),
                    "type": item.get("type", "general"),
                    "question": question,
                }
            )
    return tasks


def _call_teacher_openai(messages: list[dict[str, str]], cfg: TeacherConfig) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required for provider=openai") from exc

    client = OpenAI()
    response = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
    )
    return response.choices[0].message.content


def _call_teacher_vllm(messages: list[dict[str, str]], cfg: TeacherConfig) -> str:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests package is required for provider=vllm") from exc

    url = cfg.base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    response = requests.post(url, json=payload, timeout=cfg.timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _call_teacher(messages: list[dict[str, str]], cfg: TeacherConfig) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(cfg.retries):
        try:
            if cfg.provider == "openai":
                return _call_teacher_openai(messages, cfg)
            if cfg.provider == "vllm":
                return _call_teacher_vllm(messages, cfg)
            raise ValueError(f"Unknown provider: {cfg.provider}")
        except Exception as exc:
            last_error = exc
            time.sleep(cfg.retry_delay * (attempt + 1))
    raise RuntimeError(f"Teacher call failed after {cfg.retries} attempts: {last_error}")


def generate_reasoning_data(
    tasks: Iterable[dict[str, str]],
    cfg: TeacherConfig,
) -> List[dict[str, Any]]:
    records: List[dict[str, Any]] = []
    for idx, task in enumerate(tasks, start=1):
        question = task["question"]
        messages = [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        answer = _call_teacher(messages, cfg)
        messages.append({"role": "assistant", "content": answer})
        records.append(
            {
                "id": task.get("id", f"reasoning_{idx:03d}"),
                "messages": messages,
                "answer": answer,
                "source": "reasoning",
                "task_type": task.get("type", "general"),
            }
        )
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate vAGI training data.")
    parser.add_argument("--output", default="data/train_dataset.jsonl")
    parser.add_argument("--identity-count", type=int, default=80)
    parser.add_argument("--tasks", default=None, help="Optional JSONL of reasoning tasks.")
    parser.add_argument("--provider", choices=["openai", "vllm"], default="openai")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)
    identity_records = generate_identity_data(limit=args.identity_count)

    teacher_cfg = TeacherConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        retries=args.retries,
        retry_delay=args.retry_delay,
    )
    tasks = _load_reasoning_tasks(args.tasks)
    reasoning_records = generate_reasoning_data(tasks, teacher_cfg)

    all_records = identity_records + reasoning_records
    random.shuffle(all_records)
    _write_jsonl(Path(args.output), all_records)
    print(
        f"Wrote {len(all_records)} records "
        f"({len(identity_records)} identity, {len(reasoning_records)} reasoning) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
