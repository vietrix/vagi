"""Online Learning Module - Real-time learning during inference.

This module enables the AGI to learn continuously from its interactions,
not just from offline batch training. Key features:
- Confidence-based learning triggers
- Emergency learning for low-confidence situations
- Gradient accumulation for stable online updates
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Set
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
    """Gate that decides whether to trigger learning based on confidence.

    Issue 7.5: Uses learned attention-based aggregation instead of hardcoded
    concatenation. Each confidence source gets its own learned embedding and
    attention weight, enabling the model to dynamically focus on the most
    relevant confidence signals for triggering learning.
    """

    def __init__(self, hidden_size: int, num_confidence_sources: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_confidence_sources = num_confidence_sources

        # Issue 7.5: Learned embeddings for each confidence source
        self.confidence_embeddings = nn.Parameter(
            torch.randn(num_confidence_sources, hidden_size // 4) * 0.02
        )

        # Project confidence values to embedding space
        self.confidence_projector = nn.Linear(1, hidden_size // 4)

        # Issue 7.5: Multi-head attention for confidence aggregation
        self.attention_dim = hidden_size // 4
        self.query_proj = nn.Linear(hidden_size, self.attention_dim)
        self.key_proj = nn.Linear(hidden_size // 4, self.attention_dim)
        self.value_proj = nn.Linear(hidden_size // 4, self.attention_dim)

        # Gating network for combining attention output with hidden state
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size + self.attention_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # Learnable threshold with temperature
        self.threshold = nn.Parameter(torch.tensor(0.5))
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Optional: learned confidence source importance weights
        self.source_importance = nn.Parameter(torch.ones(num_confidence_sources))

    def forward(
        self,
        hidden_state: torch.Tensor,
        confidence_metrics: Dict[str, float]
    ) -> Tuple[bool, float]:
        """Decide whether to trigger learning using attention-based aggregation.

        Args:
            hidden_state: Model hidden state [B, H] or [B, T, H]
            confidence_metrics: Dict with keys like:
                - memory_confidence
                - reasoning_confidence
                - action_confidence
                - uncertainty

        Returns:
            should_learn: Boolean decision
            aggregated_confidence: Combined confidence score
        """
        device = hidden_state.device
        dtype = hidden_state.dtype

        # Extract confidence values in consistent order
        memory_conf = confidence_metrics.get('memory_confidence', 1.0)
        reasoning_conf = confidence_metrics.get('reasoning_confidence', 1.0)
        action_conf = confidence_metrics.get('action_confidence', 1.0)
        uncertainty = confidence_metrics.get('uncertainty', 0.0)

        # Create confidence tensor [num_sources]
        conf_values = torch.tensor(
            [memory_conf, reasoning_conf, action_conf, 1.0 - uncertainty],
            device=device, dtype=dtype
        )

        # Pool hidden state if needed
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.mean(dim=1)  # [B, H]

        batch_size = hidden_state.size(0)

        # Issue 7.5: Project confidence values and combine with learned embeddings
        # conf_values: [num_sources] -> [B, num_sources, 1]
        conf_expanded = conf_values.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)
        # Project each confidence value
        conf_projected = self.confidence_projector(conf_expanded)  # [B, num_sources, dim]

        # Combine with learned source embeddings (element-wise multiply)
        source_embeddings = self.confidence_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        conf_features = conf_projected * source_embeddings  # [B, num_sources, dim]

        # Apply learned importance weights
        importance = F.softmax(self.source_importance, dim=0)
        conf_features = conf_features * importance.view(1, -1, 1)

        # Issue 7.5: Attention-based aggregation
        # Query from hidden state, keys/values from confidence features
        query = self.query_proj(hidden_state).unsqueeze(1)  # [B, 1, dim]
        keys = self.key_proj(conf_features)  # [B, num_sources, dim]
        values = self.value_proj(conf_features)  # [B, num_sources, dim]

        # Scaled dot-product attention
        scale = self.attention_dim ** 0.5
        attn_scores = torch.bmm(query, keys.transpose(1, 2)) / scale  # [B, 1, num_sources]
        attn_weights = F.softmax(attn_scores / self.temperature, dim=-1)
        attended = torch.bmm(attn_weights, values).squeeze(1)  # [B, dim]

        # Combine attended confidence with hidden state through gating
        combined = torch.cat([hidden_state, attended], dim=-1)  # [B, H + dim]
        aggregated = self.gate_network(combined)  # [B, 1]
        aggregated_confidence = aggregated.mean().item()

        # Decision: learn if confidence is below threshold
        should_learn = aggregated_confidence < self.threshold.item()

        return should_learn, aggregated_confidence

    def get_attention_weights(
        self,
        hidden_state: torch.Tensor,
        confidence_metrics: Dict[str, float]
    ) -> torch.Tensor:
        """Get attention weights for interpretability.

        Returns weights showing which confidence sources influenced the decision.
        """
        device = hidden_state.device
        dtype = hidden_state.dtype

        conf_values = torch.tensor([
            confidence_metrics.get('memory_confidence', 1.0),
            confidence_metrics.get('reasoning_confidence', 1.0),
            confidence_metrics.get('action_confidence', 1.0),
            1.0 - confidence_metrics.get('uncertainty', 0.0)
        ], device=device, dtype=dtype)

        if hidden_state.dim() == 3:
            hidden_state = hidden_state.mean(dim=1)

        batch_size = hidden_state.size(0)
        conf_expanded = conf_values.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)
        conf_projected = self.confidence_projector(conf_expanded)
        source_embeddings = self.confidence_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        conf_features = conf_projected * source_embeddings

        query = self.query_proj(hidden_state).unsqueeze(1)
        keys = self.key_proj(conf_features)
        scale = self.attention_dim ** 0.5
        attn_scores = torch.bmm(query, keys.transpose(1, 2)) / scale
        attn_weights = F.softmax(attn_scores / self.temperature, dim=-1)

        return attn_weights.squeeze(1)  # [B, num_sources]


class PriorityQueue:
    """Sum tree with lazy updates for memory-efficient O(log N) sampling (Issue 7.4).

    Uses a segment tree (sum tree) data structure where:
    - Leaf nodes store priorities for each experience
    - Internal nodes store sum of children's priorities
    - Total priority = tree[1] (root)

    Lazy updates defer propagation until necessary, batching multiple
    updates for improved memory efficiency.
    """

    def __init__(self, capacity: int, lazy_threshold: int = 16):
        """Initialize priority queue.

        Args:
            capacity: Maximum number of items to store.
            lazy_threshold: Number of pending updates before forced propagation.
        """
        self.capacity = capacity
        self.tree_size = 2 * capacity
        self.tree = torch.zeros(self.tree_size)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0

        # Issue 7.4: Lazy update tracking
        self.lazy_threshold = lazy_threshold
        self._pending_updates: Dict[int, float] = {}  # tree_idx -> priority
        self._dirty_nodes: set = set()  # Nodes needing propagation

    def _propagate(self, idx: int):
        """Propagate priority change up the tree."""
        parent = idx // 2
        while parent >= 1:
            self.tree[parent] = self.tree[2 * parent] + self.tree[2 * parent + 1]
            parent //= 2

    def _lazy_propagate(self, idx: int):
        """Mark node for lazy propagation instead of immediate update."""
        self._dirty_nodes.add(idx)
        # Force propagation if too many pending
        if len(self._dirty_nodes) >= self.lazy_threshold:
            self._flush_lazy_updates()

    def _flush_lazy_updates(self):
        """Apply all pending lazy updates efficiently.

        Processes dirty nodes in bottom-up order to minimize redundant work.
        """
        if not self._dirty_nodes:
            return

        # Apply pending leaf updates first
        for tree_idx, priority in self._pending_updates.items():
            self.tree[tree_idx] = priority
        self._pending_updates.clear()

        # Sort dirty nodes in descending order (leaves first, then propagate up)
        sorted_nodes = sorted(self._dirty_nodes, reverse=True)

        # Track which parents we've already updated
        updated_parents: set = set()

        for idx in sorted_nodes:
            parent = idx // 2
            while parent >= 1 and parent not in updated_parents:
                self.tree[parent] = self.tree[2 * parent] + self.tree[2 * parent + 1]
                updated_parents.add(parent)
                parent //= 2

        self._dirty_nodes.clear()

    def add(self, data: Any, priority: float):
        """Add data with priority using lazy updates."""
        idx = self.write_idx + self.capacity
        self.data[self.write_idx] = data

        # Issue 7.4: Use lazy update for better memory efficiency
        self._pending_updates[idx] = priority
        self.tree[idx] = priority  # Update leaf immediately for correctness
        self._lazy_propagate(idx)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[List[Any], List[int], torch.Tensor]:
        """Sample batch using priorities - O(log N) per sample.

        Flushes lazy updates before sampling to ensure correctness.
        """
        # Ensure tree is up-to-date before sampling
        self._flush_lazy_updates()

        if self.size == 0:
            return [], [], torch.tensor([])

        indices = []
        priorities = []
        samples = []

        total_priority = self.tree[1].item()
        if total_priority <= 0:
            # Fallback to uniform sampling if all priorities are zero
            import random
            sample_indices = random.sample(range(self.size), min(batch_size, self.size))
            for idx in sample_indices:
                if self.data[idx] is not None:
                    indices.append(idx)
                    priorities.append(1.0)
                    samples.append(self.data[idx])
            weights = torch.ones(len(samples))
            return samples, indices, weights

        segment = total_priority / batch_size

        for i in range(batch_size):
            # Random point in segment (stratified sampling)
            s = segment * (i + torch.rand(1).item())

            # Find leaf via tree traversal
            idx = 1
            while idx < self.capacity:
                left = 2 * idx
                if left >= self.tree_size:
                    break
                left_sum = self.tree[left].item()
                if s <= left_sum:
                    idx = left
                else:
                    s -= left_sum
                    idx = left + 1

            data_idx = idx - self.capacity
            if 0 <= data_idx < self.size and self.data[data_idx] is not None:
                indices.append(data_idx)
                priorities.append(max(self.tree[idx].item(), 1e-8))
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
        """Update priority at index using lazy propagation."""
        tree_idx = idx + self.capacity
        self._pending_updates[tree_idx] = priority
        self.tree[tree_idx] = priority
        self._lazy_propagate(tree_idx)

    def batch_update_priorities(self, indices: List[int], priorities: List[float]):
        """Batch update multiple priorities efficiently (Issue 7.4).

        More memory-efficient than individual updates as propagation
        is deferred and batched.
        """
        for idx, priority in zip(indices, priorities):
            tree_idx = idx + self.capacity
            self._pending_updates[tree_idx] = priority
            self.tree[tree_idx] = priority
            self._dirty_nodes.add(tree_idx)

        # Single flush at the end
        self._flush_lazy_updates()

    def total_priority(self) -> float:
        """Get total priority sum (root of sum tree)."""
        self._flush_lazy_updates()
        return self.tree[1].item()


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
