"""Training script for full AGI model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.agi_config import AGIConfig, load_agi_config, load_agi_small_config
from core.agi_model import AGIModel
from core.language import BytePairTokenizer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train AGI model")
    
    parser.add_argument("--config", type=str, default="default", choices=["default", "small", "large"])
    parser.add_argument("--data-dir", type=str, default="data/text_corpus")
    parser.add_argument("--output-dir", type=str, default="checkpoints/agi")
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_name: str) -> AGIConfig:
    """Load configuration."""
    if config_name == "small":
        return load_agi_small_config()
    elif config_name == "large":
        from core.agi_config import load_agi_large_config
        return load_agi_large_config()
    else:
        return load_agi_config()


def create_text_dataset(data_dir: Path, tokenizer: BytePairTokenizer, max_length: int) -> List[torch.Tensor]:
    """Create dataset from text files."""
    dataset = []
    
    if not data_dir.exists():
        print(f"Warning: Data directory {data_dir} does not exist. Creating dummy data.")
        for i in range(100):
            text = f"This is a sample text number {i} for training the AGI model. "
            text += "It contains multiple sentences and various concepts to learn from. "
            tokens = tokenizer.encode(text, max_length=max_length)
            dataset.append(torch.tensor(tokens, dtype=torch.long))
        return dataset
    
    text_files = list(data_dir.glob("*.txt"))
    
    for text_file in text_files:
        text = text_file.read_text(encoding="utf-8")
        
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        
        for chunk in chunks:
            tokens = tokenizer.encode(chunk, max_length=max_length)
            if len(tokens) > 10:
                dataset.append(torch.tensor(tokens, dtype=torch.long))
    
    return dataset


def train_epoch(
    model: AGIModel,
    dataset: List[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch with ALL components."""
    model.train()
    
    total_loss = 0.0
    total_lm_loss = 0.0
    total_policy_loss = 0.0
    total_kg_loss = 0.0
    total_memory_loss = 0.0
    total_reasoning_loss = 0.0
    num_batches = 0
    
    device = next(model.parameters()).device
    
    indices = torch.randperm(len(dataset))
    
    for i in range(0, len(dataset), args.batch_size):
        batch_indices = indices[i:i + args.batch_size]
        batch_tokens = [dataset[idx] for idx in batch_indices]
        
        max_len = max(t.size(0) for t in batch_tokens)
        padded = torch.zeros(len(batch_tokens), max_len, dtype=torch.long)
        
        for j, tokens in enumerate(batch_tokens):
            padded[j, :tokens.size(0)] = tokens
        
        input_ids = padded.to(device)
        labels = input_ids.clone()
        
        obs = torch.randn(input_ids.size(0), model.cfg.obs_dim, device=device)
        
        state = model.init_state(batch_size=input_ids.size(0), device=device)
        
        entities = torch.randint(0, model.cfg.num_entities, (input_ids.size(0),), device=device)
        relations = torch.randint(0, model.cfg.num_relations, (input_ids.size(0),), device=device)
        
        targets = {
            "actions": torch.randint(0, model.cfg.action_dim, (input_ids.size(0),), device=device),
            "values": torch.randn(input_ids.size(0), device=device),
            "kg_triples": torch.stack([
                entities,
                relations,
                torch.randint(0, model.cfg.num_entities, (input_ids.size(0),), device=device)
            ], dim=1),
            "kg_labels": torch.ones(input_ids.size(0), device=device),
            "loss_weights": {
                "masked_lm": 1.0,
                "knowledge_graph": 0.5,
                "augmented_policy": 1.0,
                "augmented_value": 0.5,
            }
        }
        
        autocast_context = torch.autocast("cpu", dtype=torch.bfloat16) if args.use_bf16 else torch.autocast("cpu", enabled=False)
        
        with autocast_context:
            outputs = model(
                input_ids=input_ids,
                obs=obs,
                state=state,
                entities=entities,
                relations=relations,
                mode="train",
                labels=labels,
                targets=targets,
                return_loss=True
            )
            
            loss = outputs.get("loss", torch.tensor(0.0))
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: NaN/Inf loss detected, skipping batch")
                continue
            
            loss = loss / args.grad_accumulation_steps
        
        loss.backward()
        
        if (num_batches + 1) % args.grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * args.grad_accumulation_steps
        
        losses_breakdown = outputs.get("losses_breakdown", {})
        if "masked_lm" in losses_breakdown:
            total_lm_loss += losses_breakdown["masked_lm"].item()
        if "augmented_policy" in losses_breakdown:
            total_policy_loss += losses_breakdown["augmented_policy"].item()
        if "knowledge_graph" in losses_breakdown:
            total_kg_loss += losses_breakdown["knowledge_graph"].item()
        
        num_batches += 1
        
        if num_batches % args.log_every == 0:
            avg_loss = total_loss / num_batches
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch} | Batch {num_batches} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")
            if total_lm_loss > 0:
                print(f"  LM: {total_lm_loss/num_batches:.4f} | Policy: {total_policy_loss/num_batches:.4f} | KG: {total_kg_loss/num_batches:.4f}")
    
    metrics = {
        "total_loss": total_loss / max(num_batches, 1),
        "lm_loss": total_lm_loss / max(num_batches, 1),
        "policy_loss": total_policy_loss / max(num_batches, 1),
        "kg_loss": total_kg_loss / max(num_batches, 1),
    }
    
    return metrics


def save_checkpoint(
    model: AGIModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    step: int,
    output_dir: Path,
) -> None:
    """Save model checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    
    checkpoint_path = output_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    print(f"Saved checkpoint to {checkpoint_path}")


def main() -> None:
    """Main training function."""
    args = parse_args()
    
    set_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    print(f"Model parameters: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}, n_layers={config.n_layers}")
    
    tokenizer = BytePairTokenizer(vocab_size=config.vocab_size)
    
    print("Training tokenizer...")
    sample_texts = [
        "This is a sample text for tokenizer training.",
        "Machine learning and artificial intelligence are fascinating fields.",
        "Natural language processing enables computers to understand human language.",
    ]
    tokenizer.train(sample_texts, num_merges=1000)
    
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")
    
    print("Creating dataset...")
    data_dir = Path(args.data_dir)
    dataset = create_text_dataset(data_dir, tokenizer, max_length=config.max_seq_len)
    print(f"Dataset size: {len(dataset)} sequences")
    
    print("Initializing model...")
    model = AGIModel(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model moved to device: {device}")
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(dataset) // args.batch_size * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    start_epoch = 0
    global_step = 0
    
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["step"]
    
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        metrics = train_epoch(model, dataset, optimizer, scheduler, args, epoch)
        
        print(f"Epoch {epoch + 1} completed. Metrics: {metrics}")
        
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, output_dir)
        
        global_step += len(dataset) // args.batch_size
    
    final_checkpoint_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Training completed. Final model saved to {final_checkpoint_path}")


if __name__ == "__main__":
    main()
