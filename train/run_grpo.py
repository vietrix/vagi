"""Run GRPO training with Unsloth + TRL."""

from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig

from train.data_loader import SYSTEM_PROMPT, format_grpo_dataset
from train.rewards import correctness_reward, reflection_reward, xml_structure_reward

try:
    from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Unsloth is required for GRPO training. Install `unsloth`.") from exc

from trl import GRPOConfig, GRPOTrainer


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    dataset_name: str
    dataset_split: str
    dataset_config: str | None
    output_dir: str
    max_seq_length: int
    max_completion_length: int
    num_generations: int
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    logging_steps: int
    load_in_4bit: bool
    system_prompt: str


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Run GRPO training with Unsloth.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name or path.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="HuggingFace dataset name or local data file path.",
    )
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/grpo")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--system_prompt", type=str, default=SYSTEM_PROMPT)
    args = parser.parse_args()
    return TrainConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        load_in_4bit=args.load_in_4bit,
        system_prompt=args.system_prompt,
    )


def load_training_dataset(cfg: TrainConfig) -> Dataset:
    dataset_name = cfg.dataset_name
    path = Path(dataset_name)
    if path.exists():
        if path.suffix in {".json", ".jsonl"}:
            dataset = load_dataset("json", data_files=str(path), split=cfg.dataset_split)
        else:
            raise ValueError(f"Unsupported local dataset file: {path}")
    else:
        dataset = load_dataset(dataset_name, cfg.dataset_config, split=cfg.dataset_split)
    return format_grpo_dataset(dataset, system_prompt=cfg.system_prompt)


def build_model_and_tokenizer(cfg: TrainConfig) -> tuple[Any, Any]:
    if PatchFastRL is not None:
        PatchFastRL("GRPO", FastLanguageModel)

    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=dtype,
        load_in_4bit=cfg.load_in_4bit,
    )
    return model, tokenizer


def compute_max_prompt_length(cfg: TrainConfig, tokenizer: Any, dataset: Dataset) -> int:
    sample = dataset[0]
    prompt = sample["prompt"]
    if isinstance(prompt, list) and hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    elif isinstance(prompt, list):
        rendered = "\n".join(f"{msg.get('role')}: {msg.get('content')}" for msg in prompt)
    else:
        rendered = str(prompt)
    tokenized = tokenizer(rendered, add_special_tokens=True)
    prompt_length = len(tokenized["input_ids"])
    max_prompt = max(cfg.max_seq_length - cfg.max_completion_length, 1)
    return min(prompt_length, max_prompt)


def build_trainer(cfg: TrainConfig, model: Any, tokenizer: Any, dataset: Dataset) -> GRPOTrainer:
    max_prompt_length = compute_max_prompt_length(cfg, tokenizer, dataset)
    training_args = GRPOConfig(
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_generations=cfg.num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        max_steps=cfg.max_steps,
        report_to="none",
        output_dir=cfg.output_dir,
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
    return GRPOTrainer(**trainer_kwargs)


def main() -> None:
    cfg = parse_args()
    dataset = load_training_dataset(cfg)
    model, tokenizer = build_model_and_tokenizer(cfg)
    trainer = build_trainer(cfg, model, tokenizer, dataset)
    trainer.train()
    trainer.save_model(cfg.output_dir)


if __name__ == "__main__":
    main()
