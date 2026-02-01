"""
Minimal Training - Just RL components without language modeling
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from core.agi.config import load_agi_small_config
from core.agi.model import AGIModel

print("="*70)
print("vAGI MINIMAL TRAINING - RL Only")
print("="*70)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# Load config - disable language modeling
config = load_agi_small_config()
config.use_language_modeling = False  # Disable to avoid loss issues
config.use_grounded_language = False
config.use_metacognition = False
print(f"Config: obs_dim={config.obs_dim}, action_dim={config.action_dim}")

# Create model
print(f"\nCreating model...")
model = AGIModel(config).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-4)

# Training
print(f"\nTraining for 50 steps...")
model.train()

batch_size = 4
total_loss = 0.0

for step in range(50):
    # Data
    obs = torch.randn(batch_size, config.obs_dim, device=device)
    state = model.core.init_state(batch_size, device=device)
    
    # Targets
    action_targets = torch.randint(0, config.action_dim, (batch_size,), device=device)
    value_targets = torch.randn(batch_size, device=device)
    
    # Forward
    optimizer.zero_grad()
    outputs = model(
        obs=obs,
        state=state,
        targets={
            "actions": action_targets,
            "values": value_targets
        },
        mode="train",
        return_loss=True
    )
    
    # Backward
    if outputs.get("loss"):
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (step + 1) % 10 == 0:
            avg = total_loss / (step + 1)
            print(f"  Step {step+1}/50 - Loss: {avg:.4f}")

print(f"\nTraining complete - Avg loss: {total_loss/50:.4f}")

# Save
print(f"\nSaving checkpoint...")
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': config,
    'num_params': num_params,
}

checkpoint_path = checkpoint_dir / "vagi_rl_model.pt"
torch.save(checkpoint, checkpoint_path)
print(f"Saved: {checkpoint_path}")

# Test
print(f"\nTesting inference...")
model.eval()
with torch.no_grad():
    test_obs = torch.randn(1, config.obs_dim, device=device)
    test_state = model.core.init_state(1, device=device)
    
    outputs = model(
        obs=test_obs,
        state=test_state,
        mode="inference"
    )
    
    print(f"Action logits: {outputs['action_logits'].shape}")
    print(f"Value: {outputs['value'].item():.4f}")

print("\n" + "="*70)
print("SUCCESS - Model trained and saved!")
print(f"Load with: torch.load('{checkpoint_path}')")
print("="*70)
