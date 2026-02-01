"""
Quick Training Script - Train vAGI on sample data
Creates a trained checkpoint for testing AGI capabilities
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
print("vAGI QUICK TRAINING - Creating Trained Model")
print("="*70)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[1] Device: {device}")

# Load config
config = load_agi_small_config()
config.use_grounded_language = False
config.use_metacognition = False
print(f"[2] Config: vocab={config.vocab_size}, hidden={config.hidden_size}")

# Create model
print(f"\n[3] Creating model...")
model = AGIModel(config).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"    Parameters: {num_params:,}")

# Create optimizer
optimizer = AdamW(model.parameters(), lr=1e-4)
print(f"[4] Optimizer: AdamW (lr=1e-4)")

# Training data - Simple sequences
print(f"\n[5] Preparing training data...")
batch_size = 4
seq_len = 32
num_batches = 100

# Simple training loop
print(f"\n[6] Training for {num_batches} batches...")
model.train()

total_loss = 0.0
for batch_idx in range(num_batches):
    # Generate random data
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    obs = torch.randn(batch_size, config.obs_dim, device=device)
    state = model.core.init_state(batch_size, device=device)
    
    # Targets
    labels = input_ids.clone()
    targets = {
        "actions": torch.randint(0, config.action_dim, (batch_size,), device=device),
        "values": torch.randn(batch_size, device=device),
    }
    
    # Forward
    optimizer.zero_grad()
    outputs = model(
        input_ids=input_ids,
        obs=obs,
        state=state,
        labels=labels,
        targets=targets,
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
        
        if (batch_idx + 1) % 20 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"    Batch {batch_idx+1}/{num_batches} - Loss: {avg_loss:.4f}")

avg_loss = total_loss / num_batches
print(f"\n[7] Training complete - Average loss: {avg_loss:.4f}")

# Save checkpoint
print(f"\n[8] Saving checkpoint...")
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
    'num_params': num_params,
    'final_loss': avg_loss,
}

checkpoint_path = checkpoint_dir / "vagi_trained.pt"
torch.save(checkpoint, checkpoint_path)
print(f"    Saved to: {checkpoint_path}")

# Test inference
print(f"\n[9] Testing inference...")
model.eval()
with torch.no_grad():
    test_input = torch.randint(0, config.vocab_size, (1, 16), device=device)
    test_obs = torch.randn(1, config.obs_dim, device=device)
    test_state = model.core.init_state(1, device=device)
    
    outputs = model(
        input_ids=test_input,
        obs=test_obs,
        state=test_state,
        mode="inference"
    )
    
    print(f"    Output keys: {list(outputs.keys())[:5]}...")
    print(f"    Action logits: {outputs['action_logits'].shape}")
    print(f"    Value: {outputs['value'].shape}")

# Test intrinsic motivation
if hasattr(model, 'intrinsic_motivation'):
    print(f"\n[10] Testing AGI capabilities...")
    with torch.no_grad():
        state_tensor = test_obs
        action = torch.tensor([0], device=device)
        next_state = torch.randn(1, config.obs_dim, device=device)
        
        rewards = model.intrinsic_motivation.compute_intrinsic_reward(
            state=state_tensor,
            action=action,
            next_state=next_state,
            done=False
        )
        
        print(f"    Intrinsic reward: {rewards['intrinsic_reward'].item():.4f}")
        print(f"    Curiosity: {rewards['curiosity'].item():.4f}")
        print(f"    AGI modules: ACTIVE")

print("\n" + "="*70)
print("TRAINING COMPLETE - Model ready for testing!")
print(f"Checkpoint: {checkpoint_path}")
print("="*70)
