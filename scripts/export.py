#!/usr/bin/env python3
"""Merge base model + adapters and export a release checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Export vAGI release model.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapters", default="models/adapters")
    parser.add_argument("--output-dir", default="models/release")
    parser.add_argument("--output-name", default="vAGI-Core-v1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.output_name}.safetensors"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(model, args.adapters)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "name": "vAGI Core",
        "version": args.output_name,
        "creator": "Vietrix",
        "organization": "Vietrix",
        "architecture": "Sovereign Core (Moved by Logic)",
    }
    metadata_path = output_dir / "model_info.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Saved merged model to {output_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
