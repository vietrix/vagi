"""Online Learning Module - Real-time learning during inference.

This module enables the AGI to learn continuously from its interactions,
not just from offline batch training. Key features:
- Confidence-based learning triggers
- Emergency learning for low-confidence situations
- Gradient accumulation for stable online updates
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from .experience import ExperienceBuffer, ExperienceRecord


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning."""
    confidence_threshold: float = 0.5  # Learn when confidence < this
    emergency_threshold: float = 0.3  # Trigger emergency learning below this
    emergency_lr_multiplier: float = 2.0  # LR boost for emergency learning
    min_experiences: int = 4  # Min experiences before update
    max_grad_norm: float = 1.0  # Gradient clipping
    accumulation_steps: int = 4  # Gradient accumulation
    temporal_decay: float = 0.95  # Decay factor for old experiences


class ConfidenceGate(nn.Module):
    """Gate that decides whether to trigger learning based on confidence."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Confidence aggregator from multiple sources
        self.confidence_aggregator = nn.Sequential(
            nn.Linear(hidden_size + 4, hidden_size // 2),  # +4 for confidence metrics
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        hidden_state: torch.Tensor,
        confidence_metrics: Dict[str, float]
    ) -> Tuple[bool, float]:
        """Decide whether to trigger learning.

        Args:
            hidden_state: Model hidden state [B, H]
            confidence_metrics: Dict with keys like:
                - memory_confidence
                - reasoning_confidence
                - action_confidence
                - uncertainty

        Returns:
            should_learn: Boolean decision
            aggregated_confidence: Combined confidence score
        """
        # Extract confidence values
        memory_conf = confidence_metrics.get('memory_confidence', 1.0)
        reasoning_conf = confidence_metrics.get('reasoning_confidence', 1.0)
        action_conf = confidence_metrics.get('action_confidence', 1.0)
        uncertainty = confidence_metrics.get('uncertainty', 0.0)

        # Create confidence tensor
        conf_tensor = torch.tensor(
            [memory_conf, reasoning_conf, action_conf, 1.0 - uncertainty],
            device=hidden_state.device,
            dtype=hidden_state.dtype
        ).unsqueeze(0)

        # Pool hidden state if needed
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.mean(dim=1)

        # Combine with hidden state
        combined = torch.cat([hidden_state, conf_tensor], dim=-1)

        # Compute aggregated confidence
        aggregated = self.confidence_aggregator(combined)
        aggregated_confidence = aggregated.item()

        # Decision: learn if confidence is below threshold
        should_learn = aggregated_confidence < self.threshold.item()

        return should_learn, aggregated_confidence


class PriorityQueue:
    """Segment tree-based priority queue for O(log N) sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree_size = 2 * capacity
        self.tree = torch.zeros(self.tree_size)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx: int):
        """Propagate priority change up the tree."""
        parent = idx // 2
        while parent >= 1:
            self.tree[parent] = self.tree[2 * parent] + self.tree[2 * parent + 1]
            parent //= 2

    def add(self, data: Any, priority: float):
        """Add data with priority."""
        idx = self.write_idx + self.capacity
        self.data[self.write_idx] = data
        self.tree[idx] = priority
        self._propagate(idx)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[List[Any], List[int], torch.Tensor]:
        """Sample batch using priorities - O(log N) per sample."""
        if self.size == 0:
            return [], [], torch.tensor([])

        indices = []
        priorities = []
        samples = []

        total_priority = self.tree[1].item()
        segment = total_priority / batch_size

        for i in range(batch_size):
            # Random point in segment
            s = segment * (i + torch.rand(1).item())

            # Find leaf
            idx = 1
            while idx < self.capacity:
                left = 2 * idx
                if s <= self.tree[left]:
                    idx = left
                else:
                    s -= self.tree[left]
                    idx = left + 1

            data_idx = idx - self.capacity
            if data_idx < self.size and self.data[data_idx] is not None:
                indices.append(data_idx)
                priorities.append(self.tree[idx].item())
                samples.append(self.data[data_idx])

        # Compute importance sampling weights
        if priorities:
            priorities_tensor = torch.tensor(priorities)
            probs = priorities_tensor / total_priority
            weights = (self.size * probs) ** (-0.4)  # beta=0.4
            weights = weights / weights.max()
        else:
            weights = torch.tensor([])

        return samples, indices, weights

    def update_priority(self, idx: int, priority: float):
        """Update priority at index."""
        tree_idx = idx + self.capacity
        self.tree[tree_idx] = priority
        self._propagate(tree_idx)


class OnlineExperienceBuffer:
    """Experience buffer optimized for online learning."""

    def __init__(
        self,
        capacity: int = 1000,
        temporal_decay: float = 0.95
    ):
        self.capacity = capacity
        self.temporal_decay = temporal_decay
        self.priority_queue = PriorityQueue(capacity)
        self.step_counter = 0

    def add(
        self,
        experience: Dict[str, Any],
        confidence: float,
        loss: Optional[float] = None
    ):
        """Add experience with confidence-based priority."""
        # Priority = (1 - confidence) + loss_magnitude
        priority = (1.0 - confidence)
        if loss is not None:
            priority += abs(loss) * 0.5
        priority = max(priority, 0.01)  # Minimum priority

        # Add timestamp
        experience['timestamp'] = self.step_counter
        self.step_counter += 1

        self.priority_queue.add(experience, priority)

    def sample(self, batch_size: int) -> Tuple[List[Dict], torch.Tensor]:
        """Sample batch with importance weights."""
        samples, indices, weights = self.priority_queue.sample(batch_size)

        # Apply temporal decay to older samples
        if samples:
            current_step = self.step_counter
            for i, sample in enumerate(samples):
                age = current_step - sample.get('timestamp', current_step)
                decay = self.temporal_decay ** age
                weights[i] *= decay

        return samples, weights

    def update_priorities(
        self,
        indices: List[int],
        new_priorities: List[float]
    ):
        """Update priorities after learning."""
        for idx, priority in zip(indices, new_priorities):
            self.priority_queue.update_priority(idx, max(priority, 0.01))


class OnlineLearner:
    """Main online learning system for inference-time updates."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[OnlineLearningConfig] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or OnlineLearningConfig()

        # Get hidden size
        if hasattr(model, 'cfg') and hasattr(model.cfg, 'hidden_size'):
            hidden_size = model.cfg.hidden_size
        else:
            hidden_size = 512

        # Confidence gate
        self.confidence_gate = ConfidenceGate(hidden_size)

        # Experience buffer for online learning
        self.buffer = OnlineExperienceBuffer(
            capacity=1000,
            temporal_decay=self.config.temporal_decay
        )

        # Gradient accumulation
        self.accumulated_grads = 0
        self.accumulated_loss = 0.0

        # Statistics
        self.stats = {
            'online_updates': 0,
            'emergency_updates': 0,
            'skipped_updates': 0,
            'average_confidence': 1.0,
            'average_loss': 0.0
        }

        # Store original learning rate
        self._base_lr = optimizer.param_groups[0]['lr']

    def should_learn(
        self,
        hidden_state: torch.Tensor,
        confidence_metrics: Dict[str, float]
    ) -> Tuple[bool, bool, float]:
        """Determine if and how to learn.

        Returns:
            should_learn: Whether to trigger any learning
            is_emergency: Whether this is an emergency (very low confidence)
            confidence: Aggregated confidence score
        """
        should_learn, confidence = self.confidence_gate(
            hidden_state.detach(),
            confidence_metrics
        )

        is_emergency = confidence < self.config.emergency_threshold

        # Update stats
        self.stats['average_confidence'] = (
            0.95 * self.stats['average_confidence'] + 0.05 * confidence
        )

        return should_learn, is_emergency, confidence

    def observe(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        confidence: float,
        outputs: Dict[str, Any],
        info: Optional[Dict[str, Any]] = None
    ):
        """Observe an inference step for potential learning."""
        experience = {
            'state': {k: v.detach().cpu() for k, v in state.items()},
            'action': action.detach().cpu(),
            'outputs': {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in outputs.items()
                if k in ['action_logits', 'value', 'world_pred', 'uncertainty']
            },
            'info': info or {}
        }

        # Compute loss if possible
        loss = None
        if 'loss' in outputs:
            loss = outputs['loss'].item() if isinstance(outputs['loss'], torch.Tensor) else outputs['loss']

        self.buffer.add(experience, confidence, loss)

    def immediate_update(
        self,
        experience: Dict[str, Any],
        is_emergency: bool = False
    ) -> Dict[str, float]:
        """Perform immediate update from single experience.

        Used when confidence is very low and immediate learning is needed.
        """
        self.model.train()

        # Adjust learning rate for emergency
        if is_emergency:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self._base_lr * self.config.emergency_lr_multiplier

        # Prepare batch from single experience
        device = next(self.model.parameters()).device

        state = experience.get('state', {})
        obs = state.get('obs')
        if obs is not None:
            obs = obs.unsqueeze(0).to(device)

        action = experience.get('action')
        if action is not None:
            input_ids = action.unsqueeze(0).long().to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(-1)
        else:
            input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)

        # Forward pass with loss
        self.optimizer.zero_grad()

        outputs = self.model(
            input_ids=input_ids,
            obs=obs,
            return_loss=True
        )

        loss = outputs.get('loss')
        if loss is not None:
            # Clip gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()

            # Restore learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self._base_lr

            # Update stats
            if is_emergency:
                self.stats['emergency_updates'] += 1
            else:
                self.stats['online_updates'] += 1

            self.stats['average_loss'] = (
                0.9 * self.stats['average_loss'] + 0.1 * loss.item()
            )

            self.model.eval()
            return {'status': 'success', 'loss': loss.item(), 'emergency': is_emergency}

        self.model.eval()
        return {'status': 'no_loss'}

    def batch_update(self) -> Dict[str, float]:
        """Perform batch update from buffer when sufficient experiences collected."""
        if self.buffer.priority_queue.size < self.config.min_experiences:
            return {'status': 'insufficient_data'}

        self.model.train()
        device = next(self.model.parameters()).device

        # Sample batch
        samples, weights = self.buffer.sample(self.config.min_experiences)
        if not samples:
            return {'status': 'no_samples'}

        # Prepare batch
        obs_list = []
        action_list = []
        for sample in samples:
            state = sample.get('state', {})
            if 'obs' in state:
                obs_list.append(state['obs'])
            if 'action' in sample:
                action_list.append(sample['action'])

        if not obs_list:
            return {'status': 'no_obs'}

        obs = torch.stack(obs_list).to(device)

        if action_list:
            input_ids = torch.stack(action_list).long().to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(-1)
        else:
            input_ids = torch.zeros((len(samples), 1), dtype=torch.long, device=device)

        # Forward pass
        self.optimizer.zero_grad()

        outputs = self.model(
            input_ids=input_ids,
            obs=obs,
            return_loss=True
        )

        loss = outputs.get('loss')
        if loss is not None:
            # Weight by importance sampling
            weighted_loss = loss * weights.to(device).mean()

            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()

            self.stats['online_updates'] += 1
            self.stats['average_loss'] = (
                0.9 * self.stats['average_loss'] + 0.1 * weighted_loss.item()
            )

            self.model.eval()
            return {'status': 'success', 'loss': weighted_loss.item(), 'batch_size': len(samples)}

        self.model.eval()
        return {'status': 'no_loss'}

    def process_inference_step(
        self,
        hidden_state: torch.Tensor,
        confidence_metrics: Dict[str, float],
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Main entry point: process an inference step for online learning.

        This should be called after each inference forward pass.
        """
        # Decide whether to learn
        should_learn, is_emergency, confidence = self.should_learn(
            hidden_state, confidence_metrics
        )

        # Always observe (add to buffer)
        self.observe(state, action, confidence, outputs)

        result = {
            'should_learn': should_learn,
            'is_emergency': is_emergency,
            'confidence': confidence,
            'update_result': None
        }

        if not should_learn:
            self.stats['skipped_updates'] += 1
            return result

        # Emergency: immediate single-sample update
        if is_emergency:
            experience = {
                'state': state,
                'action': action,
                'outputs': outputs
            }
            result['update_result'] = self.immediate_update(experience, is_emergency=True)
        else:
            # Normal: batch update if enough experiences
            result['update_result'] = self.batch_update()

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get online learning statistics."""
        return {
            **self.stats,
            'buffer_size': self.buffer.priority_queue.size,
            'config': {
                'confidence_threshold': self.config.confidence_threshold,
                'emergency_threshold': self.config.emergency_threshold
            }
        }
