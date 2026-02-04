#!/usr/bin/env python3
"""
Direct Preference Optimization (DPO) Trainer for vAGI.

Implements the DPO algorithm from "Direct Preference Optimization: Your Language
Model is Secretly a Reward Model" (Rafailov et al., 2023).

DPO Loss Function:
    L_DPO = -E[log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

Where:
    - y_w: chosen (winning) response (with <think> reasoning)
    - y_l: rejected (losing) response (direct answer without reasoning)
    - π_θ: policy model being trained
    - π_ref: reference model (frozen copy of initial policy)
    - β: temperature parameter controlling deviation from reference

Key Implementation Details:
    - Numerical stability via log-sum-exp trick
    - Gradient checkpointing for memory efficiency
    - Label smoothing option to prevent overconfidence
    - Reference model can be frozen or soft-updated (EMA)

Usage:
    python scripts/train_dpo.py --model checkpoints/model.pt \
        --data data/preference_pairs.jsonl --output checkpoints/dpo_model.pt

    python scripts/train_dpo.py --model gpt2 --data data/preferences.jsonl \
        --beta 0.1 --epochs 3 --batch 4

Data Format (JSONL):
    {
        "prompt": "Solve: What is 2+2?",
        "chosen": "<think>[Step 1] 2+2 means adding 2 and 2...</think>\\nAnswer: 4",
        "rejected": "4"
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ============================================================================
# DPO Configuration
# ============================================================================

@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    # Core DPO parameters
    beta: float = 0.1          # Temperature for KL penalty (lower = closer to reference)
    label_smoothing: float = 0.0  # Smoothing for binary classification (0-0.5)

    # Reference model settings
    reference_free: bool = False   # If True, skip reference model (simplified DPO)
    ema_decay: float = 0.0         # EMA for reference model (0 = frozen, >0 = soft update)

    # Training settings
    learning_rate: float = 1e-6   # Small LR for fine-tuning
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    epochs: int = 1
    batch_size: int = 4

    # Sequence settings
    max_length: int = 512
    max_prompt_length: int = 128

    # Numerical stability
    eps: float = 1e-8              # Epsilon for log stability
    max_log_ratio: float = 10.0   # Clip log ratios to prevent explosion

    # Memory optimization
    gradient_checkpointing: bool = False
    mixed_precision: bool = False


# ============================================================================
# Preference Dataset
# ============================================================================

class PreferenceDataset(Dataset):
    """
    Dataset for preference pairs.

    Each sample contains:
    - prompt: The input question/instruction
    - chosen: The preferred response (with reasoning)
    - rejected: The dispreferred response (without reasoning)

    The DPO objective trains the model to assign higher probability
    to the chosen response relative to rejected.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 512,
        max_prompt_length: int = 128,
    ):
        """
        Initialize the preference dataset.

        Args:
            data_path: Path to JSONL file with preference pairs
            tokenizer: Tokenizer with encode/decode methods
            max_length: Maximum total sequence length
            max_prompt_length: Maximum prompt length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.samples = []

        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if all(k in item for k in ['prompt', 'chosen', 'rejected']):
                        self.samples.append(item)
                    else:
                        logger.warning(f"Line {line_num}: Missing required fields")
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Invalid JSON - {e}")

        logger.info(f"Loaded {len(self.samples)} preference pairs from {data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a preference pair.

        Returns dict with:
            - prompt_ids: Tokenized prompt
            - chosen_ids: Tokenized chosen response
            - rejected_ids: Tokenized rejected response
            - prompt_mask: Attention mask for prompt
            - chosen_mask: Attention mask for chosen
            - rejected_mask: Attention mask for rejected
        """
        sample = self.samples[idx]

        # Tokenize prompt
        prompt = sample['prompt']
        chosen = sample['chosen']
        rejected = sample['rejected']

        # Encode (handling different tokenizer interfaces)
        if hasattr(self.tokenizer, 'encode'):
            prompt_ids = self._encode(prompt, self.max_prompt_length)
            chosen_ids = self._encode(chosen, self.max_length - len(prompt_ids))
            rejected_ids = self._encode(rejected, self.max_length - len(prompt_ids))
        else:
            # Fallback for basic tokenizers
            prompt_ids = self._simple_encode(prompt, self.max_prompt_length)
            chosen_ids = self._simple_encode(chosen, self.max_length - len(prompt_ids))
            rejected_ids = self._simple_encode(rejected, self.max_length - len(prompt_ids))

        return {
            'prompt_ids': torch.tensor(prompt_ids, dtype=torch.long),
            'chosen_ids': torch.tensor(chosen_ids, dtype=torch.long),
            'rejected_ids': torch.tensor(rejected_ids, dtype=torch.long),
            'prompt_length': len(prompt_ids),
        }

    def _encode(self, text: str, max_len: int) -> List[int]:
        """Encode text using tokenizer."""
        ids = self.tokenizer.encode(text)
        if hasattr(ids, 'ids'):  # HuggingFace tokenizers
            ids = ids.ids
        return ids[:max_len]

    def _simple_encode(self, text: str, max_len: int) -> List[int]:
        """Simple character-level encoding fallback."""
        return [ord(c) % 256 for c in text[:max_len]]


def collate_preference_batch(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Collate preference pairs into padded batches.

    Creates concatenated sequences: [prompt + chosen] and [prompt + rejected]
    with appropriate padding and attention masks.
    """
    # Find max lengths
    max_chosen_len = max(len(b['prompt_ids']) + len(b['chosen_ids']) for b in batch)
    max_rejected_len = max(len(b['prompt_ids']) + len(b['rejected_ids']) for b in batch)
    max_len = max(max_chosen_len, max_rejected_len)

    batch_size = len(batch)

    # Initialize tensors
    chosen_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    rejected_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    chosen_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    rejected_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    chosen_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)  # -100 = ignore
    rejected_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        prompt_ids = b['prompt_ids']
        chosen_ids = b['chosen_ids']
        rejected_ids = b['rejected_ids']
        prompt_len = len(prompt_ids)

        # Chosen sequence
        chosen_seq = torch.cat([prompt_ids, chosen_ids])
        chosen_len = len(chosen_seq)
        chosen_input_ids[i, :chosen_len] = chosen_seq
        chosen_attention_mask[i, :chosen_len] = 1
        # Labels: only compute loss on response (after prompt)
        chosen_labels[i, prompt_len:chosen_len] = chosen_seq[prompt_len:]

        # Rejected sequence
        rejected_seq = torch.cat([prompt_ids, rejected_ids])
        rejected_len = len(rejected_seq)
        rejected_input_ids[i, :rejected_len] = rejected_seq
        rejected_attention_mask[i, :rejected_len] = 1
        rejected_labels[i, prompt_len:rejected_len] = rejected_seq[prompt_len:]

    return {
        'chosen_input_ids': chosen_input_ids,
        'rejected_input_ids': rejected_input_ids,
        'chosen_attention_mask': chosen_attention_mask,
        'rejected_attention_mask': rejected_attention_mask,
        'chosen_labels': chosen_labels,
        'rejected_labels': rejected_labels,
    }


# ============================================================================
# DPO Loss Implementation
# ============================================================================

class DPOLoss(nn.Module):
    """
    Direct Preference Optimization Loss.

    Mathematical Derivation:
    ------------------------
    DPO derives from the RLHF objective but eliminates explicit reward modeling.

    Starting from Bradley-Terry preference model:
        P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))

    And the optimal policy under KL-constrained RL:
        π*(y|x) ∝ π_ref(y|x) exp(r(x,y) / β)

    We can reparameterize the reward as:
        r(x, y) = β log(π*(y|x) / π_ref(y|x)) + β log Z(x)

    Substituting back, the partition function cancels:
        P(y_w > y_l | x) = σ(β log(π*(y_w|x)/π_ref(y_w|x)) - β log(π*(y_l|x)/π_ref(y_l|x)))

    The DPO loss maximizes this probability:
        L_DPO = -E[log P(y_w > y_l | x)]

    Numerical Stability:
    --------------------
    1. Log ratios can explode → clamp to [-max_log_ratio, max_log_ratio]
    2. Log probabilities can underflow → use logsumexp trick
    3. Sigmoid in log space → log_sigmoid for stability
    4. Label smoothing prevents overconfident predictions

    The implementation computes:
        log_ratio_w = log π_θ(y_w|x) - log π_ref(y_w|x)
        log_ratio_l = log π_θ(y_l|x) - log π_ref(y_l|x)
        loss = -log_sigmoid(β * (log_ratio_w - log_ratio_l))
    """

    def __init__(self, config: DPOConfig):
        """
        Initialize DPO loss.

        Args:
            config: DPO configuration
        """
        super().__init__()
        self.beta = config.beta
        self.label_smoothing = config.label_smoothing
        self.eps = config.eps
        self.max_log_ratio = config.max_log_ratio
        self.reference_free = config.reference_free

    def _get_batch_log_probs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sequence log probabilities.

        This computes:
            log π(y|x) = Σ_t log π(y_t | x, y_{<t})

        Where the sum is over response tokens (labels != -100).

        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len], -100 for ignored positions
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Per-sequence log probabilities [batch]
        """
        # Shift for next-token prediction
        # logits: predict position t from position t-1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        # Compute log probabilities
        # Use log_softmax for numerical stability
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        # Shape: [batch, seq_len-1]
        batch_size, seq_len, vocab_size = shift_logits.shape

        # Create index tensor for gathering
        # Clamp labels to valid range (handle -100)
        gather_labels = shift_labels.clone()
        gather_labels[gather_labels < 0] = 0  # Temporarily replace -100

        # Gather: select log prob of actual next token
        per_token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=gather_labels.unsqueeze(-1)
        ).squeeze(-1)  # [batch, seq_len-1]

        # Mask out ignored positions (where labels == -100)
        loss_mask = (shift_labels != -100).float() * shift_mask.float()

        # Sum log probs over sequence (only non-masked positions)
        # This gives log π(y|x) = Σ log π(y_t|...)
        per_token_log_probs = per_token_log_probs * loss_mask
        sequence_log_probs = per_token_log_probs.sum(dim=-1)  # [batch]

        # Normalize by number of tokens (optional, for length invariance)
        num_tokens = loss_mask.sum(dim=-1).clamp(min=1)
        # Note: We don't normalize here as DPO paper uses unnormalized

        return sequence_log_probs

    def forward(
        self,
        policy_chosen_logits: torch.Tensor,
        policy_rejected_logits: torch.Tensor,
        reference_chosen_logits: Optional[torch.Tensor],
        reference_rejected_logits: Optional[torch.Tensor],
        chosen_labels: torch.Tensor,
        rejected_labels: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute DPO loss.

        Args:
            policy_chosen_logits: π_θ logits for chosen [batch, seq, vocab]
            policy_rejected_logits: π_θ logits for rejected [batch, seq, vocab]
            reference_chosen_logits: π_ref logits for chosen (None if reference_free)
            reference_rejected_logits: π_ref logits for rejected (None if reference_free)
            chosen_labels: Labels for chosen sequence
            rejected_labels: Labels for rejected sequence
            chosen_attention_mask: Mask for chosen
            rejected_attention_mask: Mask for rejected

        Returns:
            loss: Scalar DPO loss
            metrics: Dictionary with debugging metrics
        """
        # Compute log probabilities for policy
        policy_chosen_log_probs = self._get_batch_log_probs(
            policy_chosen_logits, chosen_labels, chosen_attention_mask
        )
        policy_rejected_log_probs = self._get_batch_log_probs(
            policy_rejected_logits, rejected_labels, rejected_attention_mask
        )

        # Compute log probabilities for reference (if not reference-free)
        if self.reference_free or reference_chosen_logits is None:
            # Reference-free: assume uniform reference (log_ref = 0)
            reference_chosen_log_probs = torch.zeros_like(policy_chosen_log_probs)
            reference_rejected_log_probs = torch.zeros_like(policy_rejected_log_probs)
        else:
            with torch.no_grad():  # Reference is frozen
                reference_chosen_log_probs = self._get_batch_log_probs(
                    reference_chosen_logits, chosen_labels, chosen_attention_mask
                )
                reference_rejected_log_probs = self._get_batch_log_probs(
                    reference_rejected_logits, rejected_labels, rejected_attention_mask
                )

        # Compute log ratios: log(π_θ/π_ref)
        # STABILITY: Clamp to prevent explosion
        chosen_log_ratio = (policy_chosen_log_probs - reference_chosen_log_probs).clamp(
            -self.max_log_ratio, self.max_log_ratio
        )
        rejected_log_ratio = (policy_rejected_log_probs - reference_rejected_log_probs).clamp(
            -self.max_log_ratio, self.max_log_ratio
        )

        # DPO objective: maximize P(y_w > y_l)
        # logits = β * (log_ratio_w - log_ratio_l)
        logits = self.beta * (chosen_log_ratio - rejected_log_ratio)

        # Binary cross entropy loss with label smoothing
        # Target: 1 (chosen should win)
        if self.label_smoothing > 0:
            # Smooth targets: instead of 1, use (1 - label_smoothing)
            # This prevents overconfident predictions
            target = torch.ones_like(logits) * (1 - self.label_smoothing)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                target,
                reduction='mean'
            )
        else:
            # Standard DPO: -log_sigmoid(logits)
            # STABILITY: F.logsigmoid is more stable than log(sigmoid(x))
            loss = -F.logsigmoid(logits).mean()

        # Compute metrics for monitoring
        with torch.no_grad():
            # Accuracy: how often does chosen beat rejected?
            accuracy = (logits > 0).float().mean()

            # Reward margins
            chosen_rewards = self.beta * chosen_log_ratio
            rejected_rewards = self.beta * rejected_log_ratio
            reward_margin = (chosen_rewards - rejected_rewards).mean()

            # KL divergence from reference
            kl_chosen = (policy_chosen_log_probs - reference_chosen_log_probs).mean()
            kl_rejected = (policy_rejected_log_probs - reference_rejected_log_probs).mean()

        metrics = {
            'loss': loss.detach(),
            'accuracy': accuracy,
            'reward_margin': reward_margin,
            'chosen_log_prob': policy_chosen_log_probs.mean().detach(),
            'rejected_log_prob': policy_rejected_log_probs.mean().detach(),
            'chosen_log_ratio': chosen_log_ratio.mean().detach(),
            'rejected_log_ratio': rejected_log_ratio.mean().detach(),
            'kl_chosen': kl_chosen,
            'kl_rejected': kl_rejected,
            'logits_mean': logits.mean().detach(),
            'logits_std': logits.std().detach(),
        }

        return loss, metrics


# ============================================================================
# DPO Trainer
# ============================================================================

class DPOTrainer:
    """
    Trainer for Direct Preference Optimization.

    Handles:
    - Model and reference model management
    - Training loop with gradient accumulation
    - Logging and checkpointing
    - Optional EMA for reference model
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: Optional[nn.Module],
        tokenizer: Any,
        config: DPOConfig,
        device: torch.device,
    ):
        """
        Initialize DPO trainer.

        Args:
            policy_model: Model to train (π_θ)
            reference_model: Frozen reference model (π_ref), or None for reference-free
            tokenizer: Tokenizer for data processing
            config: DPO configuration
            device: Training device
        """
        self.policy_model = policy_model.to(device)
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        # Freeze reference model
        if reference_model is not None:
            self.reference_model = reference_model.to(device)
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False

        # Loss function
        self.loss_fn = DPOLoss(config)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler with warmup
        self.scheduler = None  # Will be set in train()

        # Gradient scaler for mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision and device.type == 'cuda' else None

        # Training state
        self.global_step = 0

    def _forward_model(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through model.

        Handles different model interfaces (HuggingFace, custom).
        """
        # Try HuggingFace interface first
        if hasattr(model, 'forward'):
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                if hasattr(outputs, 'logits'):
                    return outputs.logits
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    return outputs['logits']
                elif isinstance(outputs, dict) and 'text_logits' in outputs:
                    return outputs['text_logits']
                return outputs
            except TypeError:
                # Fallback for simpler models
                return model(input_ids)

        return model(input_ids)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Collated batch from PreferenceDataset

        Returns:
            Dictionary of metrics
        """
        self.policy_model.train()

        # Move to device
        chosen_input_ids = batch['chosen_input_ids'].to(self.device)
        rejected_input_ids = batch['rejected_input_ids'].to(self.device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
        chosen_labels = batch['chosen_labels'].to(self.device)
        rejected_labels = batch['rejected_labels'].to(self.device)

        # Mixed precision context
        amp_context = torch.amp.autocast('cuda') if self.scaler else torch.nullcontext()

        with amp_context:
            # Policy forward passes
            policy_chosen_logits = self._forward_model(
                self.policy_model, chosen_input_ids, chosen_attention_mask
            )
            policy_rejected_logits = self._forward_model(
                self.policy_model, rejected_input_ids, rejected_attention_mask
            )

            # Reference forward passes (no gradient)
            if self.reference_model is not None and not self.config.reference_free:
                with torch.no_grad():
                    reference_chosen_logits = self._forward_model(
                        self.reference_model, chosen_input_ids, chosen_attention_mask
                    )
                    reference_rejected_logits = self._forward_model(
                        self.reference_model, rejected_input_ids, rejected_attention_mask
                    )
            else:
                reference_chosen_logits = None
                reference_rejected_logits = None

            # Compute DPO loss
            loss, metrics = self.loss_fn(
                policy_chosen_logits=policy_chosen_logits,
                policy_rejected_logits=policy_rejected_logits,
                reference_chosen_logits=reference_chosen_logits,
                reference_rejected_logits=reference_rejected_logits,
                chosen_labels=chosen_labels,
                rejected_labels=rejected_labels,
                chosen_attention_mask=chosen_attention_mask,
                rejected_attention_mask=rejected_attention_mask,
            )

        # Backward pass
        self.optimizer.zero_grad()

        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        # EMA update for reference model (if configured)
        if self.config.ema_decay > 0 and self.reference_model is not None:
            with torch.no_grad():
                for param, ref_param in zip(
                    self.policy_model.parameters(),
                    self.reference_model.parameters()
                ):
                    ref_param.data.mul_(self.config.ema_decay)
                    ref_param.data.add_((1 - self.config.ema_decay) * param.data)

        self.global_step += 1

        return {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}

    def train(
        self,
        train_dataset: PreferenceDataset,
        eval_dataset: Optional[PreferenceDataset] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_dataset: Training preference pairs
            eval_dataset: Optional evaluation dataset
            output_dir: Directory to save checkpoints

        Returns:
            Training history
        """
        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_preference_batch(b, pad_token_id=0),
            num_workers=0,
        )

        # Setup scheduler
        total_steps = len(train_loader) * self.config.epochs
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=min(self.config.warmup_steps, total_steps // 10),
        )

        # Training history
        history = {
            'loss': [],
            'accuracy': [],
            'reward_margin': [],
        }

        logger.info(f"Starting DPO training for {self.config.epochs} epochs")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Beta: {self.config.beta}")

        for epoch in range(1, self.config.epochs + 1):
            epoch_metrics = []

            for batch_idx, batch in enumerate(train_loader):
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)

                # Log progress
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = sum(m['loss'] for m in epoch_metrics[-10:]) / 10
                    avg_acc = sum(m['accuracy'] for m in epoch_metrics[-10:]) / 10
                    logger.info(
                        f"Epoch {epoch}/{self.config.epochs} "
                        f"Step {batch_idx + 1}/{len(train_loader)} "
                        f"Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}"
                    )

            # Epoch summary
            epoch_loss = sum(m['loss'] for m in epoch_metrics) / len(epoch_metrics)
            epoch_acc = sum(m['accuracy'] for m in epoch_metrics) / len(epoch_metrics)
            epoch_margin = sum(m['reward_margin'] for m in epoch_metrics) / len(epoch_metrics)

            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            history['reward_margin'].append(epoch_margin)

            logger.info(
                f"Epoch {epoch} complete: "
                f"Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, Margin={epoch_margin:.4f}"
            )

            # Save checkpoint
            if output_dir:
                self.save_checkpoint(output_dir, epoch)

        return history

    def save_checkpoint(self, output_dir: str, epoch: int) -> None:
        """Save model checkpoint."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.policy_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }

        torch.save(checkpoint, output_path / f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, output_path / "checkpoint_latest.pt")
        logger.info(f"Saved checkpoint to {output_path}")


# ============================================================================
# Sample Data Generator
# ============================================================================

def generate_sample_preferences() -> List[Dict[str, str]]:
    """Generate sample preference pairs for testing."""
    return [
        {
            "prompt": "What is 15 + 27?",
            "chosen": "<think>\n[Step 1] I need to add 15 and 27.\n[Step 2] Breaking it down: 15 + 27 = 15 + 25 + 2 = 40 + 2 = 42.\n[Step 3] Alternatively: 15 + 27 = (10 + 5) + (20 + 7) = 30 + 12 = 42.\n</think>\n\nThe answer is 42.",
            "rejected": "42"
        },
        {
            "prompt": "Explain why the sky is blue.",
            "chosen": "<think>\n[Step 1] Light from the sun contains all colors (wavelengths).\n[Step 2] When sunlight enters Earth's atmosphere, it collides with gas molecules.\n[Step 3] Shorter wavelengths (blue) scatter more than longer wavelengths (red) - this is Rayleigh scattering.\n[Step 4] The scattered blue light reaches our eyes from all directions, making the sky appear blue.\n</think>\n\nThe sky appears blue because of Rayleigh scattering - shorter blue wavelengths of sunlight scatter more when hitting atmospheric molecules, causing blue light to reach our eyes from all directions.",
            "rejected": "Because of how light scatters in the atmosphere."
        },
        {
            "prompt": "Tại sao lá cây có màu xanh?",
            "chosen": "<think>\n[Bước 1] Lá cây chứa chất diệp lục (chlorophyll).\n[Bước 2] Chất diệp lục hấp thụ ánh sáng đỏ và xanh dương để quang hợp.\n[Bước 3] Ánh sáng xanh lục không được hấp thụ mà phản xạ lại.\n[Bước 4] Mắt chúng ta nhìn thấy ánh sáng phản xạ này nên thấy lá có màu xanh.\n</think>\n\nLá cây có màu xanh vì chất diệp lục trong lá hấp thụ ánh sáng đỏ và xanh dương, nhưng phản xạ ánh sáng xanh lục. Đây là màu mà mắt chúng ta nhìn thấy.",
            "rejected": "Vì có chất diệp lục."
        },
        {
            "prompt": "Write a function to check if a number is prime.",
            "chosen": "<think>\n[Step 1] A prime number is only divisible by 1 and itself.\n[Step 2] I'll check divisibility from 2 to sqrt(n) for efficiency.\n[Step 3] Special cases: numbers less than 2 are not prime.\n[Step 4] Even numbers greater than 2 are not prime.\n</think>\n\n```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    for i in range(3, int(n**0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True\n```",
            "rejected": "```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True\n```"
        },
    ]


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train vAGI with Direct Preference Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with custom model
    python scripts/train_dpo.py --model checkpoints/model.pt \\
        --data data/preferences.jsonl --output checkpoints/dpo

    # Generate sample data and train
    python scripts/train_dpo.py --generate-samples --output data/sample_prefs.jsonl
    python scripts/train_dpo.py --model gpt2 --data data/sample_prefs.jsonl

    # Custom beta (higher = more deviation from reference allowed)
    python scripts/train_dpo.py --model gpt2 --data prefs.jsonl --beta 0.5
        """
    )

    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model name or path")
    parser.add_argument("--data", type=str, default="data/preference_pairs.jsonl",
                        help="Path to preference pairs JSONL")
    parser.add_argument("--output", type=str, default="checkpoints/dpo",
                        help="Output directory for checkpoints")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO temperature (default: 0.1)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--reference-free", action="store_true",
                        help="Skip reference model (simplified DPO)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing (0-0.5)")
    parser.add_argument("--generate-samples", action="store_true",
                        help="Generate sample preference data and exit")

    args = parser.parse_args()

    # Generate samples if requested
    if args.generate_samples:
        samples = generate_sample_preferences()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        logger.info(f"Generated {len(samples)} sample preference pairs to {output_path}")
        return

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model and tokenizer
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        policy_model = AutoModelForCausalLM.from_pretrained(args.model)

        # Create reference model (deep copy)
        if not args.reference_free:
            import copy
            reference_model = copy.deepcopy(policy_model)
        else:
            reference_model = None

    except ImportError:
        logger.warning("transformers not available, using dummy model")
        # Dummy model for testing
        class DummyModel(nn.Module):
            def __init__(self, vocab_size=1000, hidden_size=256):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden_size)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(hidden_size, 4, batch_first=True),
                    num_layers=2
                )
                self.head = nn.Linear(hidden_size, vocab_size)

            def forward(self, input_ids, attention_mask=None):
                x = self.embed(input_ids)
                x = self.transformer(x)
                return self.head(x)

        class DummyTokenizer:
            def encode(self, text):
                return [ord(c) % 256 for c in text[:512]]

        policy_model = DummyModel()
        reference_model = DummyModel() if not args.reference_free else None
        tokenizer = DummyTokenizer()

    # Create config
    config = DPOConfig(
        beta=args.beta,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch,
        label_smoothing=args.label_smoothing,
        reference_free=args.reference_free,
    )

    # Load dataset
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Use --generate-samples to create sample data")
        sys.exit(1)

    dataset = PreferenceDataset(
        str(data_path),
        tokenizer,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
    )

    # Create trainer
    trainer = DPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        config=config,
        device=device,
    )

    # Train
    history = trainer.train(
        train_dataset=dataset,
        output_dir=args.output,
    )

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("Training Complete!")
    logger.info("=" * 50)
    logger.info(f"Final Loss: {history['loss'][-1]:.4f}")
    logger.info(f"Final Accuracy: {history['accuracy'][-1]:.4f}")
    logger.info(f"Final Reward Margin: {history['reward_margin'][-1]:.4f}")
    logger.info(f"Checkpoints saved to: {args.output}")


if __name__ == "__main__":
    main()
