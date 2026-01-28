"""Tokenizer wrapper utilities."""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer


class TokenizerWrapper:
    """Thin wrapper around Hugging Face AutoTokenizer."""

    def __init__(self, name_or_path: str, **kwargs: Any) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)

    def encode(self, text: str, **kwargs: Any):
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, ids, **kwargs: Any):
        return self.tokenizer.decode(ids, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.tokenizer, name)
