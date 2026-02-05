#!/usr/bin/env python3
"""Train vAGI with Unsloth + GRPO."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, List, Mapping

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

try:
    from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Unsloth is required. Install `unsloth`.") from exc

from train.rewards import correctness_reward, reflection_reward, xml_structure_reward


SYSTEM_PROMPT = (
    "You are vAGI, created by Vietrix. When solving problems, refer to yourself "
    "as vAGI. Use <think> tags for reasoning."
)


def _load_jsonl(path: Path) -> List[dict[str, Any]]:
    records: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def _normalize_conversations(conversations: Iterable[Mapping[str, Any]]) -> List[dict[str, str]]:
    messages: List[dict[str, str]] = []
    for item in conversations:
        role = str(item.get("role") or item.get("from") or "").lower()
        content = str(item.get("content") or item.get("value") or "")
        if not content:
            continue
        if role in {"human", "user"}:
            role = "user"
        elif role in {"assistant", "gpt", "bot"}:
            role = "assistant"
        elif role != "system":
            role = "user"
        messages.append({"role": role, "content": content})
    return messages


def _ensure_system_prompt(messages: List[dict[str, str]]) -> List[dict[str, str]]:
    if not messages:
        raise ValueError("Empty message list after normalization")
    if messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return messages


def _extract_answer(messages: List[dict[str, str]], fallback: str) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return fallback


def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    messages: List[dict[str, str]]
    if "messages" in record:
        messages = _normalize_conversations(record["messages"])
    elif "conversations" in record:
        messages = _normalize_conversations(record["conversations"])
    elif "prompt" in record:
        messages = _normalize_conversations(record["prompt"])
    else:
        user_text = record.get("input") or record.get("question") or record.get("prompt") or ""
        assistant_text = record.get("output") or record.get("answer") or ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(user_text)},
            {"role": "assistant", "content": str(assistant_text)},
        ]
    messages = _ensure_system_prompt(messages)
    answer = record.get("answer") or _extract_answer(messages, "")
    return {"prompt": messages, "answer": str(answer)}


def _build_dataset(records: Iterable[Mapping[str, Any]]) -> Dataset:
    normalized = [_normalize_record(record) for record in records]
    if not normalized:
        raise ValueError("No training records found in dataset.")
    return Dataset.from_list(normalized)


def _compute_max_steps(dataset_size: int, cfg: argparse.Namespace) -> int:
    if cfg.max_steps is not None:
        return cfg.max_steps
    steps_per_epoch = math.ceil(
        dataset_size / (cfg.batch_size * cfg.gradient_accumulation_steps)
    )
    return max(1, steps_per_epoch * cfg.epochs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train vAGI with GRPO.")
    parser.add_argument("--data", default="data/train_dataset.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", default="models/adapters")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    records = _load_jsonl(data_path)
    dataset = _build_dataset(records)

    PatchFastRL("GRPO", FastLanguageModel)
    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
    )

    max_prompt_length = max(args.max_seq_length - args.max_completion_length, 1)
    max_steps = _compute_max_steps(len(dataset), args)

    training_args = GRPOConfig(
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=max_steps,
        report_to="none",
        output_dir=args.output_dir,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "reward_funcs": [correctness_reward, xml_structure_reward, reflection_reward],
        "train_dataset": dataset,
        "peft_config": peft_config,
    }
    signature = inspect.signature(GRPOTrainer.__init__)
    if "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
