#!/usr/bin/env python3
"""
Chat with vAGI model - RAW neural output only.

Usage:
    python scripts/chat.py
    python scripts/chat.py --model checkpoints/model.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stdin.reconfigure(encoding='utf-8', errors='replace')

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agi import AGIModel
from core.agi.config import load_agi_small_config
from core.nlp import BytePairTokenizer


def generate(model, tokenizer, prompt: str, device, max_tokens=200, temp=0.8, show_reasoning=True):
    """Generate text từ model - RAW output."""

    # Tokenize prompt
    ids = tokenizer.encode(prompt, max_length=256)
    generated = ids.copy()

    if show_reasoning:
        print(f"\n[Reasoning]")
        print(f"  Input: \"{prompt}\"")
        print(f"  Tokens: {len(ids)}")

    model.eval()
    start_time = time.time()

    # Metrics from model
    confidence = 0.5
    metacog_info = {}
    online_learning_info = {}

    with torch.no_grad():
        for step in range(max_tokens):
            x = torch.tensor([generated[-256:]], dtype=torch.long, device=device)

            # Forward pass - lấy tất cả outputs từ AGI model
            out = model(input_ids=x, mode='inference')

            # Extract metacognition info (if available)
            if 'metacognition' in out and step == 0:
                meta = out['metacognition']
                if isinstance(meta, dict):
                    confidence = meta.get('confidence', 0.5)
                    metacog_info = {
                        'confidence': confidence,
                        'uncertainty': meta.get('uncertainty', 0.0),
                        'should_think_more': meta.get('should_think_more', False),
                    }

            # Extract online learning info (if available)
            if 'online_learning' in out and step == 0:
                ol = out['online_learning']
                if isinstance(ol, dict):
                    online_learning_info = {
                        'should_learn': ol.get('should_learn', False),
                        'learning_confidence': ol.get('confidence', 0.5),
                    }

            # Get logits for next token
            logits = out.get('text_logits')
            if logits is None:
                break

            # Sample next token
            next_logits = logits[0, -1] / temp
            probs = torch.softmax(next_logits, dim=-1)

            # Top-k sampling for better quality
            top_k = 50
            top_probs, top_indices = torch.topk(probs, min(top_k, probs.size(-1)))
            top_probs = top_probs / top_probs.sum()  # renormalize

            idx = torch.multinomial(top_probs, 1).item()
            next_tok = top_indices[idx].item()

            if next_tok == 0:  # EOS/PAD
                break

            generated.append(next_tok)

    elapsed = time.time() - start_time
    result = tokenizer.decode(generated[len(ids):]).strip()

    if show_reasoning:
        print(f"  Generated: {len(generated) - len(ids)} tokens in {elapsed:.2f}s")
        if metacog_info:
            print(f"  MetaCognition: confidence={metacog_info.get('confidence', 0):.2%}")
        if online_learning_info:
            print(f"  OnlineLearning: should_learn={online_learning_info.get('should_learn', False)}")
        print()

    return result, {
        'confidence': confidence,
        'metacognition': metacog_info,
        'online_learning': online_learning_info,
        'tokens_generated': len(generated) - len(ids),
        'time': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description='Chat với vAGI - RAW output')
    parser.add_argument('--model', default='checkpoints/model.pt')
    parser.add_argument('--temp', type=float, default=0.8, help='Temperature (0.1-1.5)')
    parser.add_argument('--max-tokens', type=int, default=200, help='Max tokens to generate')
    parser.add_argument('--no-reasoning', action='store_true', help='Hide reasoning output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    tokenizer = None
    if os.path.exists(args.model):
        print(f"Loading {args.model}...")
        ckpt = torch.load(args.model, map_location=device, weights_only=False)
        config = ckpt.get('config', load_agi_small_config())
        model = AGIModel(config)
        model.load_state_dict(ckpt['model_state_dict'])
        params = sum(p.numel() for p in model.parameters())
        print(f"Loaded: {params:,} parameters")

        # Load tokenizer from checkpoint
        if 'tokenizer_vocab' in ckpt:
            tokenizer = BytePairTokenizer(vocab_size=config.vocab_size)
            tokenizer.vocab = ckpt['tokenizer_vocab']
            tokenizer.merges = [tuple(m) for m in ckpt.get('tokenizer_merges', [])]
            tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
            print(f"Loaded tokenizer: {len(tokenizer.vocab)} tokens")
    else:
        print(f"No checkpoint found at {args.model}")
        print("Creating new model (untrained)...")
        config = load_agi_small_config()
        model = AGIModel(config)

    model = model.to(device).eval()
    if tokenizer is None:
        tokenizer = BytePairTokenizer(vocab_size=config.vocab_size)

    print("\n" + "=" * 50)
    print("  vAGI Chat - RAW Neural Output")
    print("  All responses are generated by the model itself")
    print("=" * 50)
    print(f"  Device: {device}")
    print(f"  Temperature: {args.temp}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  MetaCognition: {config.use_metacognition}")
    print(f"  Online Learning: {config.online_learning_enabled}")
    print("=" * 50)
    print("  Commands: /exit, /temp <value>, /stats")
    print("=" * 50 + "\n")

    show_reasoning = not args.no_reasoning

    while True:
        try:
            user = input("Bạn: ").strip()
            if not user:
                continue

            if user == '/exit':
                print("Tạm biệt!")
                break

            if user.startswith('/temp '):
                try:
                    args.temp = float(user.split()[1])
                    print(f"Temperature set to {args.temp}")
                except:
                    print("Usage: /temp <value> (e.g., /temp 0.7)")
                continue

            if user == '/stats':
                p = sum(x.numel() for x in model.parameters())
                print(f"\n[Model Stats]")
                print(f"  Parameters: {p:,}")
                print(f"  Hidden size: {config.hidden_size}")
                print(f"  Layers: {config.n_layers}")
                print(f"  Heads: {config.n_heads}")
                print(f"  Vocab size: {config.vocab_size}")
                print(f"  Device: {device}")
                print()
                continue

            if user == '/reasoning':
                show_reasoning = not show_reasoning
                print(f"Reasoning display: {'ON' if show_reasoning else 'OFF'}")
                continue

            # Generate response từ model
            result, metrics = generate(
                model, tokenizer, user, device,
                max_tokens=args.max_tokens,
                temp=args.temp,
                show_reasoning=show_reasoning
            )

            # Display RAW output
            print(f"vAGI: {result}")
            print()

        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break
        except EOFError:
            print("\nTạm biệt!")
            break
        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == '__main__':
    main()
