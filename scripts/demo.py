#!/usr/bin/env python3
"""
Demo các capabilities của vAGI model.

Usage:
    python scripts/demo.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agi import AGIModel
from core.agi.config import load_agi_small_config


def main():
    print("=" * 50)
    print("  vAGI Demo")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create model
    print("\n[1] Creating model...")
    config = load_agi_small_config()
    model = AGIModel(config).to(device).eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {params:,}")
    print(f"    Hidden size: {config.hidden_size}")
    print(f"    Layers: {config.n_layers}")

    # Test forward
    print("\n[2] Testing forward pass...")
    try:
        with torch.no_grad():
            x = torch.randint(0, config.vocab_size, (1, 16), device=device)
            obs = torch.randn(1, config.obs_dim, device=device)
            out = model(input_ids=x, obs=obs, mode='inference')

        print(f"    Output keys: {len(out)} keys")
        print(f"    Action shape: {out.get('action_logits', torch.tensor([])).shape}")
        print("    [OK]")
    except Exception as e:
        print(f"    [FAIL] {e}")

    # Check modules
    print("\n[3] AGI Modules:")
    modules = [
        ('metacognition', 'Meta-Cognition'),
        ('program_synthesizer', 'Program Synthesis'),
        ('hierarchical_memory', 'Hierarchical Memory'),
        ('abstract_reasoner', 'Abstract Reasoning'),
        ('intrinsic_motivation', 'Intrinsic Motivation'),
        ('scene_graph_builder', 'Scene Graphs'),
    ]
    for attr, name in modules:
        has = hasattr(model, attr) and getattr(model, attr) is not None
        status = "Yes" if has else "No"
        print(f"    {name}: {status}")

    # Test configs
    print("\n[4] Config flags:")
    flags = ['use_metacognition', 'use_grounded_language', 'online_learning_enabled']
    for f in flags:
        val = getattr(config, f, 'N/A')
        print(f"    {f}: {val}")

    print("\n" + "=" * 50)
    print("Demo complete!")


if __name__ == '__main__':
    main()
