"""Simple test - model creation and forward pass only."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from core.agi.config import load_agi_small_config
from core.agi.model import AGIModel

print("="*60)
print("SIMPLE TEST - Model Creation and Forward Pass")
print("="*60)

# Load config
print("\n[1] Creating model...")
config = load_agi_small_config()
model = AGIModel(config)
print(f"  SUCCESS - Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Test forward
print("\n[2] Testing forward pass...")
batch_size = 2
seq_len = 10

input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
obs = torch.randn(batch_size, config.obs_dim)
state = model.core.init_state(batch_size)

try:
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            obs=obs,
            state=state,
            mode="inference"
        )
    
    print(f"  SUCCESS - Forward pass completed")
    print(f"  Output keys: {list(outputs.keys())[:5]}...")
    print(f"  Action logits: {outputs['action_logits'].shape}")
    
except Exception as e:
    print(f"  FAILED - {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
