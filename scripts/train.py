#!/usr/bin/env python3
"""
Train vAGI model với JSONL dataset.

Usage:
    python scripts/train.py --data data/train_dataset.jsonl --epochs 10
    python scripts/train.py --data data/train_dataset.jsonl --epochs 5 --small
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Fix output buffering
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agi import AGIModel
from core.agi.config import AGIConfig, load_agi_small_config, load_agi_tiny_config
from core.nlp import BytePairTokenizer


class TextDataset(Dataset):
    """Simple JSONL dataset."""

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

        print(f"Loaded {len(self.samples)} samples", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.samples[idx], max_length=self.max_len)
        if len(ids) < self.max_len:
            ids = ids + [0] * (self.max_len - len(ids))
        return torch.tensor(ids[:self.max_len], dtype=torch.long)


def collate(batch):
    return torch.stack(batch)


def main():
    parser = argparse.ArgumentParser(description='Train vAGI')
    parser.add_argument('--data', default='data/train_dataset.jsonl')
    parser.add_argument('--output', default='checkpoints/model.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--small', action='store_true', help='Use small config')
    parser.add_argument('--tiny', action='store_true', help='Use tiny config (fast CPU)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Config & Model
    if args.tiny:
        config = load_agi_tiny_config()
    elif args.small:
        config = load_agi_small_config()
    else:
        config = AGIConfig()
    tokenizer = BytePairTokenizer(vocab_size=config.vocab_size)

    print("Loading data...", flush=True)

    # Load raw texts for tokenizer training
    raw_texts = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    text = item.get('input', '') + ' ' + item.get('output', '')
                    raw_texts.append(text)
                except:
                    pass

    # Train tokenizer on dataset
    print(f"Training tokenizer on {len(raw_texts)} texts...", flush=True)
    tokenizer.train(raw_texts, num_merges=1000)
    print(f"Tokenizer vocab size: {len(tokenizer.vocab)}", flush=True)

    dataset = TextDataset(args.data, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=collate)

    print("Creating model...", flush=True)
    model = AGIModel(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}", flush=True)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Train
    print(f"\nTraining for {args.epochs} epochs...", flush=True)
    model.train()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            out = model(input_ids=batch, mode='train')

            # Language modeling loss
            logits = out.get('text_logits')
            if logits is None:
                continue

            # Next token prediction - align shapes
            seq_len = batch.size(1)
            pred = logits[:, :seq_len-1, :].contiguous()
            target = batch[:, 1:].contiguous()

            loss = nn.CrossEntropyLoss(ignore_index=0)(
                pred.view(-1, pred.size(-1)),
                target.view(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg:.4f}", flush=True)

    # Save model and tokenizer
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer_vocab': tokenizer.vocab,
        'tokenizer_merges': tokenizer.merges,
    }, args.output)
    print(f"\nSaved to {args.output}", flush=True)


if __name__ == '__main__':
    main()
