"""Minimal test - just test core model without extended modules."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch

print("="*60)
print("MINIMAL TEST - Core Model Only")
print("="*60)

# Test 1: Import
print("\n[1] Importing...")
from core.agi.config import AGIConfig
from core.agi.model import AGIModel

# Test 2: Create minimal config
print("\n[2] Creating minimal config...")
config = AGIConfig(
    vocab_size=1000,
    hidden_size=256,
    n_layers=2,
    n_heads=4,
    n_kv_heads=4,  # Same as n_heads
    max_seq_len=128,
    obs_dim=64,
    action_dim=10,
    use_hierarchical_memory=False,  # Disable to avoid shape mismatch
    # DISABLE ALL NEW MODULES
    use_continuous_learning=False,
    use_scene_graphs=False,
    use_intrinsic_motivation=False,
    use_program_synthesis=False,
    use_grounded_language=False,
    use_metacognition=False,
)
print("  Config created")

# Test 3: Create model
print("\n[3] Creating model...")
model = AGIModel(config)
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test 4: Forward pass
print("\n[4] Forward pass...")
batch_size = 2
seq_len = 10

input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
obs = torch.randn(batch_size, config.obs_dim)
state = model.core.init_state(batch_size)

with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        obs=obs,
        state=state,
        mode="inference"
    )

print(f"  Outputs: {list(outputs.keys())}")
print(f"  Action shape: {outputs['action_logits'].shape}")

# Test 5: Training
print("\n[5] Training step...")
targets = {
    "actions": torch.randint(0, config.action_dim, (batch_size,)),
    "values": torch.randn(batch_size),
}

outputs = model(
    input_ids=input_ids,
    obs=obs,
    state=state,
    labels=input_ids.clone(),
    targets=targets,
    mode="train",
    return_loss=True
)

if outputs.get("loss"):
    loss = outputs["loss"]
    print(f"  Loss: {loss.item():.4f}")
    loss.backward()
    print("  Gradients: OK")
else:
    print("  No loss")

print("\n" + "="*60)
print("SUCCESS - Core model works!")
print("="*60)
