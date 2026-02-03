#!/usr/bin/env python3
"""Prepare and tokenize training data for vAGI.

This script converts raw text/experience data into the format needed for training.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any

from transformers import AutoTokenizer


def load_tokenizer(model_name: str = "gpt2"):
    """Load tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_text_data(
    input_file: str,
    output_file: str,
    tokenizer,
    max_seq_len: int = 1024,
    obs_dim: int = 256,
):
    """Prepare text data for language modeling."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                # Try to parse as JSON first
                data = json.loads(line)
                text = data.get("text", data.get("content", ""))
            except json.JSONDecodeError:
                # Treat as plain text
                text = line

            if not text:
                continue

            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_seq_len)

            # Create training sample
            sample = {
                "input_ids": tokens[:-1] if len(tokens) > 1 else tokens,
                "labels": tokens[1:] if len(tokens) > 1 else tokens,
                "obs": [random.gauss(0, 1) for _ in range(obs_dim)],  # Random obs for now
                "action": 0,
                "reward": 0.0,
            }

            fout.write(json.dumps(sample) + "\n")
            count += 1

    print(f"Prepared {count} samples -> {output_file}")
    return count


def prepare_experience_data(
    input_file: str,
    output_file: str,
    tokenizer,
    max_seq_len: int = 1024,
    obs_dim: int = 256,
):
    """Prepare RL experience data."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # Extract fields
            text = data.get("text", data.get("instruction", ""))
            obs = data.get("obs", data.get("observation", []))
            action = data.get("action", 0)
            reward = data.get("reward", 0.0)

            # Tokenize text
            if text:
                tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_seq_len)
            else:
                tokens = [tokenizer.pad_token_id] * 10

            # Ensure obs has correct dimension
            if len(obs) < obs_dim:
                obs = obs + [0.0] * (obs_dim - len(obs))
            elif len(obs) > obs_dim:
                obs = obs[:obs_dim]

            sample = {
                "input_ids": tokens,
                "labels": tokens,
                "obs": obs,
                "action": action if isinstance(action, int) else 0,
                "reward": float(reward),
            }

            fout.write(json.dumps(sample) + "\n")
            count += 1

    print(f"Prepared {count} experience samples -> {output_file}")
    return count


def generate_synthetic_data(
    output_file: str,
    tokenizer,
    num_samples: int = 10000,
    max_seq_len: int = 1024,
    obs_dim: int = 256,
):
    """Generate synthetic training data for testing."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sample prompts for synthetic data
    prompts = [
        "The agent should",
        "To complete this task",
        "The observation shows",
        "Based on the current state",
        "The optimal action is",
        "Given the environment",
        "The reward signal indicates",
        "Processing the input",
        "Analyzing the situation",
        "The system responds by",
    ]

    completions = [
        "moving forward to reach the goal.",
        "analyzing the surroundings carefully.",
        "selecting the best action available.",
        "updating its internal state.",
        "learning from this experience.",
        "adjusting its strategy accordingly.",
        "exploring new possibilities.",
        "maximizing the expected reward.",
        "minimizing potential risks.",
        "balancing exploration and exploitation.",
    ]

    with open(output_file, 'w', encoding='utf-8') as fout:
        for i in range(num_samples):
            # Generate random text
            prompt = random.choice(prompts)
            completion = random.choice(completions)
            text = f"{prompt} {completion}"

            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_seq_len)

            sample = {
                "input_ids": tokens[:-1] if len(tokens) > 1 else tokens,
                "labels": tokens[1:] if len(tokens) > 1 else tokens,
                "obs": [random.gauss(0, 1) for _ in range(obs_dim)],
                "action": random.randint(0, 255),
                "reward": random.gauss(0, 1),
            }

            fout.write(json.dumps(sample) + "\n")

    print(f"Generated {num_samples} synthetic samples -> {output_file}")
    return num_samples


def main():
    parser = argparse.ArgumentParser(description="Prepare vAGI training data")

    parser.add_argument("--input", type=str, help="Input file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output file")
    parser.add_argument("--format", type=str, default="text",
                        choices=["text", "experience", "synthetic"])
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--obs-dim", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of samples for synthetic data")

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"Vocab size: {len(tokenizer)}")

    if args.format == "synthetic":
        generate_synthetic_data(
            args.output,
            tokenizer,
            num_samples=args.num_samples,
            max_seq_len=args.max_seq_len,
            obs_dim=args.obs_dim,
        )
    elif args.format == "text":
        prepare_text_data(
            args.input,
            args.output,
            tokenizer,
            max_seq_len=args.max_seq_len,
            obs_dim=args.obs_dim,
        )
    elif args.format == "experience":
        prepare_experience_data(
            args.input,
            args.output,
            tokenizer,
            max_seq_len=args.max_seq_len,
            obs_dim=args.obs_dim,
        )


if __name__ == "__main__":
    main()
