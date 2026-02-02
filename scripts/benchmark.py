#!/usr/bin/env python3
"""
Benchmark vAGI model performance.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --model checkpoints/model.pt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agi import AGIModel
from core.agi.config import load_agi_small_config, load_agi_tiny_config
from core.nlp import BytePairTokenizer


def benchmark_inference_speed(model, tokenizer, device, num_runs=10, max_tokens=50):
    """Benchmark inference speed."""
    prompts = [
        "Xin chào",
        "Viết hàm tính giai thừa",
        "Thủ đô Việt Nam là gì?",
        "Giải thích Machine Learning",
        "Python là ngôn ngữ lập trình",
    ]

    times = []
    tokens_generated = []

    model.eval()
    with torch.no_grad():
        for _ in range(num_runs):
            prompt = prompts[_ % len(prompts)]
            ids = tokenizer.encode(prompt, max_length=128)
            generated = ids.copy()

            start = time.time()
            for _ in range(max_tokens):
                x = torch.tensor([generated[-128:]], dtype=torch.long, device=device)
                out = model(input_ids=x, mode='inference')
                logits = out.get('text_logits')
                if logits is None:
                    break
                next_tok = logits[0, -1].argmax().item()
                if next_tok == 0:
                    break
                generated.append(next_tok)

            elapsed = time.time() - start
            times.append(elapsed)
            tokens_generated.append(len(generated) - len(ids))

    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)
    tokens_per_sec = avg_tokens / avg_time if avg_time > 0 else 0

    return {
        'avg_time_sec': round(avg_time, 3),
        'avg_tokens': round(avg_tokens, 1),
        'tokens_per_sec': round(tokens_per_sec, 1),
        'runs': num_runs,
    }


def benchmark_forward_pass(model, device, batch_sizes=[1, 2, 4, 8], seq_len=128):
    """Benchmark forward pass latency."""
    results = {}
    model.eval()

    with torch.no_grad():
        for batch_size in batch_sizes:
            # Warmup
            x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
            for _ in range(3):
                model(input_ids=x, mode='inference')

            # Benchmark
            times = []
            for _ in range(10):
                x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
                start = time.time()
                model(input_ids=x, mode='inference')
                times.append(time.time() - start)

            avg = sum(times) / len(times)
            results[f'batch_{batch_size}'] = {
                'avg_latency_ms': round(avg * 1000, 2),
                'throughput_samples_per_sec': round(batch_size / avg, 2),
            }

    return results


def benchmark_memory(model, device):
    """Benchmark memory usage."""
    if device.type != 'cuda':
        return {'note': 'Memory benchmarks require CUDA'}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Forward pass
    x = torch.randint(0, 1000, (1, 256), device=device)
    model(input_ids=x, mode='inference')

    return {
        'peak_memory_mb': round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
        'current_memory_mb': round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
    }


def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'total_formatted': f'{total:,}',
        'trainable_formatted': f'{trainable:,}',
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark vAGI')
    parser.add_argument('--model', default='checkpoints/model.pt')
    parser.add_argument('--output', default='benchmark_results.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'PyTorch version: {torch.__version__}')
    print()

    # Load model
    if os.path.exists(args.model):
        print(f'Loading checkpoint: {args.model}')
        ckpt = torch.load(args.model, map_location=device, weights_only=False)
        config = ckpt.get('config', load_agi_tiny_config())
        model = AGIModel(config)
        model.load_state_dict(ckpt['model_state_dict'])

        tokenizer = BytePairTokenizer(vocab_size=config.vocab_size)
        if 'tokenizer_vocab' in ckpt:
            tokenizer.vocab = ckpt['tokenizer_vocab']
            tokenizer.merges = [tuple(m) for m in ckpt.get('tokenizer_merges', [])]
            tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
    else:
        print('No checkpoint found, using random weights')
        config = load_agi_tiny_config()
        model = AGIModel(config)
        tokenizer = BytePairTokenizer(vocab_size=config.vocab_size)

    model = model.to(device).eval()

    # Run benchmarks
    results = {
        'device': str(device),
        'pytorch_version': torch.__version__,
        'config': {
            'hidden_size': config.hidden_size,
            'n_layers': config.n_layers,
            'n_heads': config.n_heads,
            'vocab_size': config.vocab_size,
        }
    }

    print('=' * 60)
    print('BENCHMARK RESULTS')
    print('=' * 60)

    # Parameter count
    print('\n1. Model Parameters')
    params = count_parameters(model)
    results['parameters'] = params
    print(f'   Total: {params["total_formatted"]}')
    print(f'   Trainable: {params["trainable_formatted"]}')

    # Forward pass latency
    print('\n2. Forward Pass Latency')
    latency = benchmark_forward_pass(model, device, batch_sizes=[1, 2, 4])
    results['forward_pass'] = latency
    for key, val in latency.items():
        print(f'   {key}: {val["avg_latency_ms"]}ms ({val["throughput_samples_per_sec"]} samples/sec)')

    # Inference speed
    print('\n3. Text Generation Speed')
    gen_speed = benchmark_inference_speed(model, tokenizer, device, num_runs=5, max_tokens=30)
    results['generation'] = gen_speed
    print(f'   Avg time: {gen_speed["avg_time_sec"]}s')
    print(f'   Avg tokens: {gen_speed["avg_tokens"]}')
    print(f'   Tokens/sec: {gen_speed["tokens_per_sec"]}')

    # Memory (GPU only)
    if device.type == 'cuda':
        print('\n4. Memory Usage')
        memory = benchmark_memory(model, device)
        results['memory'] = memory
        print(f'   Peak: {memory["peak_memory_mb"]} MB')
        print(f'   Current: {memory["current_memory_mb"]} MB')

    print('\n' + '=' * 60)

    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved to: {args.output}')


if __name__ == '__main__':
    main()
