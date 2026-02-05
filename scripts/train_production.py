#!/usr/bin/env python3
"""Production GRPO/SFT training script using Unsloth."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List, Mapping

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported, add_new_tokens
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Unsloth is required. Install `unsloth`.") from exc


SYSTEM_PROMPT = (
    "You are vAGI, created by Vietrix. When solving problems, refer to yourself "
    "as vAGI. Use <think> tags for reasoning."
)


def _retry(fn, *, retries: int = 3, base_delay: float = 1.0):
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            time.sleep(base_delay * (attempt + 1))
    raise RuntimeError(f"Operation failed after {retries} attempts: {last_error}")


def _load_jsonl(path: Path) -> List[dict[str, Any]]:
    records: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def _normalize_messages(conversations: Iterable[Mapping[str, Any]]) -> List[dict[str, str]]:
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


def _render_text(messages: List[dict[str, str]], tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)


def _normalize_record(record: Mapping[str, Any], tokenizer) -> dict[str, str]:
    if "messages" in record:
        messages = _normalize_messages(record["messages"])
    elif "conversations" in record:
        messages = _normalize_messages(record["conversations"])
    elif "prompt" in record:
        messages = _normalize_messages(record["prompt"])
    else:
        user_text = record.get("input") or record.get("question") or record.get("prompt") or ""
        assistant_text = record.get("output") or record.get("answer") or ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(user_text)},
            {"role": "assistant", "content": str(assistant_text)},
        ]
    messages = _ensure_system_prompt(messages)
    return {"text": _render_text(messages, tokenizer)}


def _build_dataset(records: Iterable[Mapping[str, Any]], tokenizer) -> Dataset:
    normalized = [_normalize_record(record, tokenizer) for record in records]
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


# Optional W&B logging:
# import os
# os.environ["WANDB_PROJECT"] = "vagi-training"


# Optional GGUF export (requires llama.cpp or compatible tooling):
# def merge_and_save_gguf(output_dir: str, gguf_path: str) -> None:
#     """Merge LoRA and export to GGUF for local runtimes."""
#     raise NotImplementedError("GGUF export must be implemented with a local toolchain.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train vAGI with Unsloth.")
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="models/release")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    records = _load_jsonl(data_path)

    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

    def _load_model():
        return FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            dtype=dtype,
            load_in_4bit=args.load_in_4bit,
        )

    model, tokenizer = _retry(_load_model, retries=3, base_delay=2.0)
    add_new_tokens(model, tokenizer, new_tokens=["<think>", "</think>"])

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=args.max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )

    dataset = _build_dataset(records, tokenizer)
    max_steps = _compute_max_steps(len(dataset), args)

    # learning_rate=2e-4: stable LoRA default on 1-3B models for fast convergence without overshooting.
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        output_dir=args.output_dir,
        optim="adamw_8bit",
        save_strategy="no",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
    )

    try:
        trainer.train()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
        raise

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapters + tokenizer (adapter_model.safetensors, adapter_config.json, tokenizer.json)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved adapters to {output_dir}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
