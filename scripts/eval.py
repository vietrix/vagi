#!/usr/bin/env python3
"""
Evaluate vAGI model trên dataset.

Usage:
    python scripts/eval.py --model checkpoints/model.pt --data data/train_dataset.jsonl
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agi import AGIModel
from core.agi.config import load_agi_small_config
from core.nlp import BytePairTokenizer


class EvalDataset(Dataset):
    """Dataset for evaluation."""

    def __init__(self, path: str, tokenizer, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        text = item.get('input', '') + ' ' + item.get('output', '')
                        self.samples.append(text)
                    except:
                        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.samples[idx], max_length=self.max_len)
        if len(ids) < self.max_len:
            ids = ids + [0] * (self.max_len - len(ids))
        return torch.tensor(ids[:self.max_len], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description='Evaluate vAGI')
    parser.add_argument('--model', default='checkpoints/model.pt')
    parser.add_argument('--data', default='data/train_dataset.jsonl')
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        return

    print(f"Loading {args.model}...")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    config = ckpt.get('config', load_agi_small_config())
    model = AGIModel(config).to(device).eval()
    model.load_state_dict(ckpt['model_state_dict'])

    tokenizer = BytePairTokenizer(vocab_size=config.vocab_size)
    dataset = EvalDataset(args.data, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch)

    print(f"Samples: {len(dataset)}")

    # Evaluate
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(input_ids=batch, mode='inference')

            logits = out.get('text_logits')
            if logits is not None:
                seq_len = batch.size(1)
                pred = logits[:, :seq_len-1, :].contiguous()
                target = batch[:, 1:].contiguous()
                loss = nn.CrossEntropyLoss(ignore_index=0)(
                    pred.view(-1, pred.size(-1)),
                    target.view(-1)
                )
                total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    perplexity = math.exp(avg_loss)

    print(f"\nResults:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")


if __name__ == '__main__':
    main()
