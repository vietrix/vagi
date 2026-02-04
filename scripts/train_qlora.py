#!/usr/bin/env python3
"""
QLoRA Training Script for vAGI.

Fine-tunes vAGI to utilize <think> and <verify_code> tools effectively
using 4-bit quantization for consumer GPU training.

Features:
- 4-bit NF4 quantization via bitsandbytes
- LoRA adapters targeting attention layers
- paged_adamw_32bit optimizer for memory efficiency
- Reasoning chain dataset format

Requirements:
    pip install transformers peft bitsandbytes accelerate datasets trl

Usage:
    # Basic training
    python scripts/train_qlora.py --model meta-llama/Llama-2-7b-hf --data data/reasoning.jsonl

    # With custom LoRA config
    python scripts/train_qlora.py --model ./models/vagi-7b --lora-r 64 --lora-alpha 16

    # Resume from checkpoint
    python scripts/train_qlora.py --model ./models/vagi-7b --resume checkpoints/qlora/checkpoint-1000

Hardware Requirements:
    - Minimum: 8GB VRAM (RTX 3070, RTX 4060)
    - Recommended: 16GB VRAM (RTX 4080, A4000)
    - With gradient checkpointing: 6GB VRAM possible
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
from datasets import load_dataset, Dataset

# Check for required packages
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
        PeftModel,
    )
    from trl import SFTTrainer, SFTConfig
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install transformers peft bitsandbytes accelerate datasets trl")
    sys.exit(1)


# =============================================================================
# System Prompt Template
# =============================================================================

SYSTEM_PROMPT = """You are vAGI (Genesis Edition), a reasoning-first AI system.

When solving complex problems:
1. Use <think> tags to show your reasoning process
2. Use <verify_code language="python">...</verify_code> to verify logic with code
3. Read <observation>...</observation> tags for code execution results

Example format:
<think>
Let me analyze this step by step:
1. First, I need to understand the problem
2. Then, I'll verify my logic with code
</think>

<verify_code language="python">
# Calculate the result
result = 2 + 2
print(f"Result: {result}")
</verify_code>

<observation>Result: 4</observation>

Based on the verification, the answer is 4.

Be concise, accurate, and always verify complex calculations."""


# =============================================================================
# Dataset Formatting
# =============================================================================

def format_reasoning_example(example: Dict[str, Any], tokenizer) -> str:
    """
    Format a reasoning example for training.

    Expected input format:
    {
        "instruction": "Calculate the sum of first 10 prime numbers",
        "thinking": "Let me find the first 10 primes...",
        "code": "primes = [2,3,5,7,11,13,17,19,23,29]\nprint(sum(primes))",
        "observation": "Result: 129",
        "response": "The sum of the first 10 prime numbers is 129."
    }
    """
    # Build the conversation
    messages = []

    # System message
    messages.append({
        "role": "system",
        "content": SYSTEM_PROMPT,
    })

    # User instruction
    messages.append({
        "role": "user",
        "content": example.get("instruction", example.get("input", "")),
    })

    # Assistant response with thinking and verification
    response_parts = []

    # Add thinking if present
    thinking = example.get("thinking", example.get("thought", ""))
    if thinking:
        response_parts.append(f"<think>\n{thinking}\n</think>")

    # Add code verification if present
    code = example.get("code", example.get("verify_code", ""))
    if code:
        response_parts.append(f'<verify_code language="python">\n{code}\n</verify_code>')

        # Add observation
        observation = example.get("observation", example.get("result", ""))
        if observation:
            if not observation.startswith("<observation>"):
                observation = f"<observation>{observation}</observation>"
            response_parts.append(observation)

    # Add final response
    final_response = example.get("response", example.get("output", ""))
    if final_response:
        response_parts.append(final_response)

    messages.append({
        "role": "assistant",
        "content": "\n\n".join(response_parts),
    })

    # Format using tokenizer's chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        # Fallback format
        text = f"### System:\n{SYSTEM_PROMPT}\n\n"
        text += f"### User:\n{example.get('instruction', example.get('input', ''))}\n\n"
        text += f"### Assistant:\n{'\n\n'.join(response_parts)}"
        return text


def create_formatting_func(tokenizer):
    """Create the formatting function for SFTTrainer."""
    def formatting_func(examples):
        texts = []
        for i in range(len(examples.get("instruction", examples.get("input", [])))):
            example = {k: v[i] for k, v in examples.items()}
            text = format_reasoning_example(example, tokenizer)
            texts.append(text)
        return texts
    return formatting_func


def load_reasoning_dataset(data_path: str) -> Dataset:
    """
    Load reasoning chain dataset from JSONL file.

    Expected format per line:
    {"instruction": "...", "thinking": "...", "code": "...", "observation": "...", "response": "..."}
    """
    if data_path.endswith('.jsonl'):
        return load_dataset('json', data_files=data_path, split='train')
    elif data_path.endswith('.json'):
        return load_dataset('json', data_files=data_path, split='train')
    else:
        # Try loading as HuggingFace dataset
        return load_dataset(data_path, split='train')


# =============================================================================
# Model Loading
# =============================================================================

def create_bnb_config(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "bfloat16",
    bnb_4bit_use_double_quant: bool = True,
) -> BitsAndBytesConfig:
    """Create BitsAndBytes quantization config for 4-bit loading."""
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def create_lora_config(
    r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
) -> LoraConfig:
    """
    Create LoRA configuration for attention layer adaptation.

    Args:
        r: LoRA rank (higher = more parameters, better quality)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout for LoRA layers
        target_modules: Which modules to apply LoRA to
        bias: Bias training strategy ("none", "all", "lora_only")
    """
    if target_modules is None:
        # Target all attention projection layers for comprehensive reasoning adaptation
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # Optionally include FFN layers for deeper adaptation
            # "gate_proj",
            # "up_proj",
            # "down_proj",
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_and_tokenizer(
    model_path: str,
    bnb_config: BitsAndBytesConfig,
    device_map: str = "auto",
    trust_remote_code: bool = True,
):
    """Load model with 4-bit quantization and tokenizer."""
    print(f"Loading model from {model_path}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        padding_side="right",
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

    return model, tokenizer


# =============================================================================
# Training
# =============================================================================

def create_training_args(
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_steps: int = -1,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    save_steps: int = 100,
    eval_steps: int = 100,
    max_seq_length: int = 2048,
    fp16: bool = False,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    optim: str = "paged_adamw_32bit",
) -> TrainingArguments:
    """
    Create training arguments optimized for consumer GPUs.

    Key optimizations:
    - paged_adamw_32bit: Saves ~30% optimizer memory
    - gradient_checkpointing: Trades compute for memory
    - bf16: Better numerical stability than fp16
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy="steps" if eval_steps > 0 else "no",
        eval_steps=eval_steps if eval_steps > 0 else None,
        save_total_limit=3,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=optim,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=0.3,
        group_by_length=True,
        report_to="none",  # Set to "wandb" for W&B logging
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )


def train_qlora(
    model_path: str,
    data_path: str,
    output_dir: str,
    lora_r: int = 64,
    lora_alpha: int = 16,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    resume_from_checkpoint: Optional[str] = None,
):
    """
    Main QLoRA training function.

    Args:
        model_path: Path to base model
        data_path: Path to JSONL training data
        output_dir: Output directory for checkpoints
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    print("=" * 60)
    print("QLoRA Training for vAGI")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create configs
    bnb_config = create_bnb_config()
    lora_config = create_lora_config(r=lora_r, lora_alpha=lora_alpha)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, bnb_config)

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"\nLoading dataset from {data_path}...")
    dataset = load_reasoning_dataset(data_path)
    print(f"Dataset size: {len(dataset)} examples")

    # Create formatting function
    formatting_func = create_formatting_func(tokenizer)

    # Create training arguments
    training_args = create_training_args(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        packing=False,  # Set True for efficiency with short examples
    )

    # Train
    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model()

    # Save LoRA adapter separately
    adapter_dir = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print("\nTraining complete!")
    print(f"  Full model: {output_dir}")
    print(f"  LoRA adapter: {adapter_dir}")

    return trainer


# =============================================================================
# Inference Helper
# =============================================================================

def load_qlora_for_inference(
    base_model_path: str,
    adapter_path: str,
    device_map: str = "auto",
):
    """
    Load a QLoRA-trained model for inference.

    Args:
        base_model_path: Path to original base model
        adapter_path: Path to LoRA adapter
        device_map: Device mapping

    Returns:
        model, tokenizer
    """
    # Load base model (can be 4-bit for inference too)
    bnb_config = create_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)

    # Optionally merge for faster inference
    # model = model.merge_and_unload()

    model.eval()

    return model, tokenizer


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="QLoRA Training for vAGI")

    # Model and data
    parser.add_argument("--model", type=str, required=True, help="Path to base model")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL training data")
    parser.add_argument("--output", type=str, default="checkpoints/qlora", help="Output directory")

    # LoRA config
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")

    # Training config
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")

    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    # Verify CUDA
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Training will be slow.")

    # Print config
    print("\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}")
    print(f"  Output: {args.output}")
    print(f"  LoRA rank: {args.lora_r}")
    print(f"  LoRA alpha: {args.lora_alpha}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max sequence length: {args.max_seq_length}")
    if args.resume:
        print(f"  Resuming from: {args.resume}")
    print()

    # Train
    train_qlora(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        resume_from_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
