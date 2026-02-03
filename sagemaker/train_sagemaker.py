#!/usr/bin/env python3
"""SageMaker Training Script for vAGI.

This script is designed to run on AWS SageMaker training instances.
It handles distributed training, checkpointing, and S3 integration.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agi.config import AGIConfig, load_agi_tiny_config, load_agi_small_config
from core.agi.model import AGIModel
from core.training import ContinuousLearner, ContinuousLearningConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Configurations for Different Sizes
# =============================================================================

def get_model_config(size: str) -> AGIConfig:
    """Get model configuration by size."""

    if size == "tiny":
        # 3.5M params - for testing
        return load_agi_tiny_config()

    elif size == "small":
        # ~30M params - for demos
        return AGIConfig(
            vocab_size=32000,
            hidden_size=512,
            n_layers=12,
            n_heads=8,
            n_kv_heads=4,
            mlp_ratio=4.0,
            max_seq_len=1024,
            obs_dim=256,
            obs_tokens=8,
            action_dim=256,
            memory_slots=16,
            dropout=0.1,
            use_rotary=True,
            use_gqa=True,
            use_flash_attn=True,
            use_grad_checkpoint=True,
        )

    elif size == "medium":
        # ~100M params - balanced
        return AGIConfig(
            vocab_size=50000,
            hidden_size=1024,
            n_layers=24,
            n_heads=16,
            n_kv_heads=4,
            mlp_ratio=4.0,
            max_seq_len=2048,
            obs_dim=512,
            obs_tokens=16,
            action_dim=512,
            memory_slots=32,
            dropout=0.1,
            use_rotary=True,
            use_gqa=True,
            use_flash_attn=True,
            use_grad_checkpoint=True,
        )

    elif size == "large":
        # ~350M params - production
        return AGIConfig(
            vocab_size=100000,
            hidden_size=1536,
            n_layers=32,
            n_heads=24,
            n_kv_heads=6,
            mlp_ratio=4.0,
            max_seq_len=4096,
            obs_dim=768,
            obs_tokens=32,
            action_dim=768,
            memory_slots=64,
            dropout=0.1,
            use_rotary=True,
            use_gqa=True,
            use_flash_attn=True,
            use_grad_checkpoint=True,
        )

    elif size == "xlarge":
        # ~1B params - full capability
        return AGIConfig(
            vocab_size=100000,
            hidden_size=2048,
            n_layers=48,
            n_heads=32,
            n_kv_heads=8,
            mlp_ratio=4.0,
            max_seq_len=8192,
            obs_dim=1024,
            obs_tokens=64,
            action_dim=1024,
            memory_slots=128,
            dropout=0.1,
            use_rotary=True,
            use_gqa=True,
            use_flash_attn=True,
            use_grad_checkpoint=True,
        )

    else:
        raise ValueError(f"Unknown model size: {size}")


# =============================================================================
# Dataset
# =============================================================================

class VAGIDataset(torch.utils.data.Dataset):
    """Dataset for vAGI training."""

    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 1024,
        obs_dim: int = 256,
    ):
        self.max_seq_len = max_seq_len
        self.obs_dim = obs_dim
        self.data = []

        # Load data from jsonl files
        data_path = Path(data_dir)
        for jsonl_file in data_path.glob("*.jsonl"):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        self.data.append(item)
                    except json.JSONDecodeError:
                        continue

        logger.info(f"Loaded {len(self.data)} samples from {data_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Get input_ids (tokenized text)
        input_ids = item.get("input_ids", [0] * 10)
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        else:
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))

        # Get observation
        obs = item.get("obs", [0.0] * self.obs_dim)
        if len(obs) < self.obs_dim:
            obs = obs + [0.0] * (self.obs_dim - len(obs))
        elif len(obs) > self.obs_dim:
            obs = obs[:self.obs_dim]

        # Get labels (for language modeling)
        labels = item.get("labels", input_ids.copy())
        if len(labels) > self.max_seq_len:
            labels = labels[:self.max_seq_len]
        else:
            labels = labels + [-100] * (self.max_seq_len - len(labels))

        # Get action targets
        action = item.get("action", 0)
        reward = item.get("reward", 0.0)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "obs": torch.tensor(obs, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "action": torch.tensor(action, dtype=torch.long),
            "reward": torch.tensor(reward, dtype=torch.float32),
        }


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_steps = 0

    for step, batch in enumerate(dataloader):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        obs = batch["obs"].to(device)
        labels = batch["labels"].to(device)

        # Initialize state
        batch_size = input_ids.size(0)
        state = model.module.core.init_state(batch_size, device) if hasattr(model, 'module') else model.core.init_state(batch_size, device)

        # Forward pass with mixed precision
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=args.fp16):
            outputs = model(
                input_ids=input_ids,
                obs=obs,
                state=state,
                labels=labels,
                mode="train",
                return_loss=True,
            )

            loss = outputs.get("loss")
            if loss is None:
                # Compute language modeling loss manually
                from core.training.losses import language_loss
                loss = language_loss(outputs["text_logits"], labels)

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_steps += 1

        # Logging
        if step % args.log_interval == 0:
            avg_loss = total_loss / total_steps
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch} | Step {step}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | LR: {lr:.2e}"
            )

            if writer is not None:
                global_step = epoch * len(dataloader) + step
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/avg_loss", avg_loss, global_step)
                writer.add_scalar("train/lr", lr, global_step)

    return {"loss": total_loss / max(total_steps, 1)}


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    args: argparse.Namespace,
    metrics: Dict[str, float],
):
    """Save training checkpoint."""
    model_to_save = model.module if hasattr(model, 'module') else model

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": model_to_save.cfg.__dict__,
    }

    # Save to model directory (will be uploaded to S3)
    checkpoint_path = Path(args.model_dir) / f"checkpoint-epoch-{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Also save latest
    latest_path = Path(args.model_dir) / "checkpoint-latest.pt"
    torch.save(checkpoint, latest_path)


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train vAGI on SageMaker")

    # SageMaker environment variables
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./output"))
    parser.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "./data"))
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"))

    # Model configuration
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["tiny", "small", "medium", "large", "xlarge"])

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", default=True)

    # Distributed training
    parser.add_argument("--local-rank", type=int, default=-1)

    # Logging
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1)

    args = parser.parse_args()

    # Setup distributed training
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        is_main = rank == 0
    else:
        world_size = 1
        rank = 0
        is_main = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}, Rank: {rank}/{world_size}")

    # Create model
    logger.info(f"Creating {args.model_size} model...")
    config = get_model_config(args.model_size)
    model = AGIModel(config)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {num_params / 1e6:.2f}M parameters")

    model = model.to(device)

    # Wrap with DDP if distributed
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Create dataset and dataloader
    dataset = VAGIDataset(
        args.train_dir,
        max_seq_len=config.max_seq_len,
        obs_dim=config.obs_dim,
    )

    if args.local_rank != -1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Scheduler
    total_steps = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=args.warmup_steps / total_steps,
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler(enabled=args.fp16)

    # TensorBoard writer
    writer = None
    if is_main:
        log_dir = Path(args.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)

    # Training loop
    logger.info("Starting training...")

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        metrics = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            args=args,
            writer=writer,
        )

        logger.info(f"Epoch {epoch} completed. Loss: {metrics['loss']:.4f}")

        # Save checkpoint
        if is_main and (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, args, metrics)

    # Save final model
    if is_main:
        model_to_save = model.module if hasattr(model, 'module') else model
        final_path = Path(args.model_dir) / "model_final.pt"
        torch.save(model_to_save.state_dict(), final_path)
        logger.info(f"Saved final model to {final_path}")

        # Save config
        config_path = Path(args.model_dir) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2)

    if writer is not None:
        writer.close()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
