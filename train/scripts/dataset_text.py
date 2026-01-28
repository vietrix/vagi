"""Simple text dataset with a lightweight tokenizer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


@dataclass
class SimpleTokenizer:
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    pad_id: int
    unk_id: int
    bos_id: int
    eos_id: int

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str, max_length: int | None = None) -> List[int]:
        tokens = text.strip().split()
        ids = [self.bos_id] + [self.token_to_id.get(tok, self.unk_id) for tok in tokens] + [self.eos_id]
        if max_length is not None:
            ids = ids[:max_length]
        return ids


def build_tokenizer(texts: Iterable[str], min_freq: int = 1) -> SimpleTokenizer:
    counts: Dict[str, int] = {}
    for text in texts:
        for tok in text.strip().split():
            counts[tok] = counts.get(tok, 0) + 1
    vocab = SPECIAL_TOKENS[:]
    for tok, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if count >= min_freq:
            vocab.append(tok)
    token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
    return SimpleTokenizer(
        token_to_id=token_to_id,
        id_to_token=vocab,
        pad_id=token_to_id["<pad>"],
        unk_id=token_to_id["<unk>"],
        bos_id=token_to_id["<bos>"],
        eos_id=token_to_id["<eos>"],
    )


def load_texts(path: str | Path) -> List[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".jsonl":
        texts = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if "text" not in record:
                raise ValueError("JSONL records must contain a 'text' field")
            texts.append(str(record["text"]))
        return texts
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TextDataset(Dataset):
    """Dataset that returns tokenized sequences."""

    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = [self.tokenizer.encode(text, max_length=max_length) for text in texts]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = torch.tensor(self.samples[idx], dtype=torch.long)
        return {"input_ids": ids}
