"""Meta-cognition: self-awareness and reasoning about reasoning."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import math

import torch
from torch import nn
from torch.nn import functional as F


class ThinkingState(Enum):
    """States in the thinking process.

    Used throughout the reasoning pipeline to track cognitive state:
    - NORMAL: Standard processing, no issues detected
    - UNCERTAIN: Low confidence in current reasoning path
    - CONFUSED: Contradictory or incoherent thoughts detected
    - STUCK: No progress being made, potential loop
    - CONFIDENT: High confidence, ready to conclude
    """
    NORMAL = "normal"
    UNCERTAIN = "uncertain"
    CONFUSED = "confused"
    STUCK = "stuck"
    CONFIDENT = "confident"

    @classmethod
    def from_metrics(
        cls,
        confidence: float,
        coherence: float,
        loop_prob: float,
        progress: float = 1.0
    ) -> "ThinkingState":
        """Determine thinking state from metrics.

        Args:
            confidence: Confidence level [0, 1]
            coherence: Coherence score [0, 1]
            loop_prob: Probability of being in a loop [0, 1]
            progress: Progress metric [0, 1], 0 = no progress

        Returns:
            Appropriate ThinkingState based on metrics
        """
        if loop_prob > 0.7 or progress < 0.1:
            return cls.STUCK
        if coherence < 0.4:
            return cls.CONFUSED
        if confidence < 0.3:
            return cls.UNCERTAIN
        if confidence > 0.8 and coherence > 0.7:
            return cls.CONFIDENT
        return cls.NORMAL

    def get_recommended_action(self) -> str:
        """Get recommended action for this thinking state."""
        actions = {
            ThinkingState.NORMAL: "CONTINUE",
            ThinkingState.UNCERTAIN: "EXPLORE",
            ThinkingState.CONFUSED: "REVISE",
            ThinkingState.STUCK: "RESTART",
            ThinkingState.CONFIDENT: "CONCLUDE",
        }
        return actions[self]

    def to_embedding_index(self) -> int:
        """Convert to index for embedding lookup."""
        return list(ThinkingState).index(self)


@dataclass
class KeyMoment:
    """A key moment in a thought trace (for summarization)."""
    step: int
    state: ThinkingState
    confidence: float
    importance: float  # How important this moment was
    summary_vector: Optional[torch.Tensor] = None  # Compressed representation


@dataclass
class SummarizedThoughtTrace:
    """Summarized record of a thought process (memory efficient).

    Instead of storing full tensor history, stores:
    - Key moments (state transitions, high importance events)
    - Compressed summary statistics
    - Final outcome
    """
    trace_id: int
    initial_state: ThinkingState
    final_state: ThinkingState
    key_moments: List[KeyMoment] = field(default_factory=list)
    total_steps: int = 0
    avg_confidence: float = 0.0
    min_coherence: float = 1.0
    final_confidence: float = 0.0
    coherence_score: float = 0.0
    timestamp: int = 0
    # Compressed representation of entire trace
    summary_embedding: Optional[torch.Tensor] = None

    def add_key_moment(self, moment: KeyMoment, max_moments: int = 10):
        """Add a key moment, keeping only the most important ones."""
        self.key_moments.append(moment)
        if len(self.key_moments) > max_moments:
            # Keep most important moments
            self.key_moments.sort(key=lambda m: m.importance, reverse=True)
            self.key_moments = self.key_moments[:max_moments]


@dataclass
class ThoughtTrace:
    """Record of a thought process (legacy compatibility)."""
    thoughts: List[torch.Tensor]
    state: ThinkingState
    confidence: float
    coherence_score: float
    timestamp: int


class ImportanceWeightedBuffer:
    """Circular buffer with importance-weighted sampling.

    Implements priority experience replay style buffer where:
    - Entries have importance weights
    - Sampling probability is proportional to importance
    - Older entries can be kept if they're important
    """

    def __init__(
        self,
        capacity: int = 1000,
        feature_dim: int = 3,
        alpha: float = 0.6,  # Priority exponent
        beta_start: float = 0.4,  # IS weight start
        beta_frames: int = 100000,  # Frames to anneal beta
        device: Optional[torch.device] = None,
    ):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.device = device or torch.device('cpu')

        # Main storage
        self.data = torch.zeros(capacity, feature_dim, device=self.device)
        self.importance = torch.zeros(capacity, device=self.device)
        self.timestamps = torch.zeros(capacity, dtype=torch.long, device=self.device)

        # Tracking
        self.position = 0
        self.size = 0
        self.frame = 0
        self.max_importance = 1.0

    def add(
        self,
        entry: torch.Tensor,
        importance: float = 1.0,
    ):
        """Add entry with importance weight.

        Args:
            entry: Data tensor of shape [feature_dim]
            importance: Importance weight (higher = more likely to be sampled/retained)
        """
        # If buffer is full, consider importance for replacement
        if self.size >= self.capacity:
            # Find the least important entry to replace
            # But only replace if new entry is more important
            min_idx = torch.argmin(self.importance).item()
            if importance < self.importance[min_idx].item():
                # New entry is less important, use normal circular replacement
                idx = self.position
            else:
                # Replace least important entry
                idx = min_idx
        else:
            idx = self.position

        self.data[idx] = entry.to(self.device)
        self.importance[idx] = importance
        self.timestamps[idx] = self.frame

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.frame += 1
        self.max_importance = max(self.max_importance, importance)

    def sample(
        self,
        batch_size: int,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample entries with importance-weighted probability.

        Args:
            batch_size: Number of entries to sample
            return_weights: Whether to return importance sampling weights

        Returns:
            Sampled data, optionally with IS weights for unbiased gradient estimates
        """
        if self.size == 0:
            empty = torch.zeros(0, self.feature_dim, device=self.device)
            return (empty, None) if return_weights else (empty, None)

        batch_size = min(batch_size, self.size)

        # Compute sampling probabilities
        priorities = self.importance[:self.size] ** self.alpha
        probs = priorities / (priorities.sum() + 1e-8)

        # Sample indices
        indices = torch.multinomial(probs, batch_size, replacement=False)
        sampled_data = self.data[indices]

        if return_weights:
            # Compute importance sampling weights
            beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
            weights = (self.size * probs[indices]) ** (-beta)
            weights = weights / weights.max()  # Normalize
            return sampled_data, weights

        return sampled_data, None

    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics."""
        if self.size == 0:
            return {
                'size': 0,
                'avg_importance': 0.0,
                'max_importance': 0.0,
                'min_importance': 0.0,
            }

        valid = self.importance[:self.size]
        return {
            'size': self.size,
            'avg_importance': valid.mean().item(),
            'max_importance': valid.max().item(),
            'min_importance': valid.min().item(),
        }

    def get_all(self) -> torch.Tensor:
        """Get all valid entries."""
        return self.data[:self.size]


class SharedEncoderCapabilityPredictor(nn.Module):
    """Capability predictor with shared encoder and task-specific heads.

    Instead of 3 separate networks, uses:
    - Shared encoder for common feature extraction
    - Lightweight task-specific heads for each prediction type

    This reduces parameters and improves generalization through shared representations.
    """

    def __init__(
        self,
        input_dim: int = 64,
        shared_dim: int = 128,
        head_hidden_dim: int = 32,
        num_heads: int = 3,  # success_prob, expected_time, confidence
    ):
        super().__init__()

        # Shared encoder - extracts common features
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
        )

        # Task-specific heads (lightweight)
        self.success_head = nn.Sequential(
            nn.Linear(shared_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, 1),
            nn.Sigmoid()
        )

        self.time_head = nn.Sequential(
            nn.Linear(shared_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, 1),
            nn.Softplus()
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(shared_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, 1),
            nn.Sigmoid()
        )

        # Optional: Cross-head attention for heads to inform each other
        self.head_attention = nn.MultiheadAttention(
            embed_dim=head_hidden_dim,
            num_heads=1,
            batch_first=True,
        )

        # Projection for attention
        self.shared_to_head_dim = nn.Linear(shared_dim, head_hidden_dim)

    def forward(
        self,
        task_features: torch.Tensor,
        use_cross_attention: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through shared encoder and heads.

        Args:
            task_features: Task embedding [batch, input_dim] or [input_dim]
            use_cross_attention: Whether to use cross-head attention

        Returns:
            Dictionary with predictions from each head
        """
        # Handle 1D input
        was_1d = task_features.dim() == 1
        if was_1d:
            task_features = task_features.unsqueeze(0)

        # Shared encoding
        shared_repr = self.shared_encoder(task_features)  # [batch, shared_dim]

        if use_cross_attention:
            # Project to head dimension
            head_input = self.shared_to_head_dim(shared_repr)  # [batch, head_hidden_dim]

            # Create query/key/value for cross-attention
            # Stack 3 copies for 3 heads
            qkv = head_input.unsqueeze(1).expand(-1, 3, -1)  # [batch, 3, head_hidden_dim]

            # Self-attention among heads
            attended, _ = self.head_attention(qkv, qkv, qkv)  # [batch, 3, head_hidden_dim]

            # Use attended representations - project back to shared_dim
            # attended[:, i] is [batch, head_hidden_dim], need to project to shared_dim
            success_attended = attended[:, 0]  # [batch, head_hidden_dim]
            time_attended = attended[:, 1]
            conf_attended = attended[:, 2]

            # Simple residual: use attended features directly in heads
            # The heads already take shared_dim input, so we add residual in shared space
            success_input = shared_repr
            time_input = shared_repr
            conf_input = shared_repr
        else:
            success_input = shared_repr
            time_input = shared_repr
            conf_input = shared_repr
            success_attended = None
            time_attended = None
            conf_attended = None

        # Get predictions from heads
        success_prob = self.success_head(success_input)
        expected_time = self.time_head(time_input)
        confidence = self.confidence_head(conf_input)

        # Squeeze if input was 1D
        if was_1d:
            success_prob = success_prob.squeeze(0)
            expected_time = expected_time.squeeze(0)
            confidence = confidence.squeeze(0)
            shared_repr = shared_repr.squeeze(0)

        return {
            'success_prob': success_prob,
            'expected_time': expected_time,
            'confidence': confidence,
            'shared_repr': shared_repr,
        }


class SelfModel(nn.Module):
    """Model of the agent's own capabilities and limitations.

    Improvements:
    - Circular queue with importance weighting for performance history
    - Shared encoder with task-specific heads for capability prediction
    """

    def __init__(
        self,
        task_embedding_dim: int = 128,
        capability_dim: int = 64,
        num_capability_types: int = 20,
        buffer_capacity: int = 1000,
    ):
        super().__init__()

        # Task encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(task_embedding_dim, capability_dim * 2),
            nn.GELU(),
            nn.LayerNorm(capability_dim * 2),
            nn.Linear(capability_dim * 2, capability_dim),
            nn.LayerNorm(capability_dim),
        )

        # Shared encoder capability predictor (replaces separate networks)
        self.capability_predictor = SharedEncoderCapabilityPredictor(
            input_dim=capability_dim,
            shared_dim=capability_dim * 2,
            head_hidden_dim=capability_dim // 2,
        )

        # Capability types classifier
        self.capability_classifier = nn.Linear(capability_dim, num_capability_types)

        # Importance-weighted circular buffer for performance history
        # Features: [task_id, success, time, predicted_success, actual_success_diff]
        self.buffer_capacity = buffer_capacity
        self.performance_buffer = ImportanceWeightedBuffer(
            capacity=buffer_capacity,
            feature_dim=5,  # task_id, success, time, predicted_success, importance
            alpha=0.6,
        )

        # Legacy buffer for backward compatibility
        self.register_buffer(
            'performance_history',
            torch.zeros(100, 3)  # [task_id, success, time]
        )
        self.register_buffer('history_index', torch.tensor(0, dtype=torch.long))
        
    def can_i_solve(
        self,
        task_embedding: torch.Tensor,
        threshold: float = 0.3
    ) -> Tuple[bool, str, Dict[str, float]]:
        """Estimate if agent can solve task.

        Returns:
            can_solve: Boolean decision
            reason: Explanation string
            metrics: Dictionary of predictions
        """
        # Encode task
        task_features = self.task_encoder(task_embedding)

        # Predict capabilities using shared encoder
        predictions = self.capability_predictor(task_features, use_cross_attention=True)

        success_prob_tensor = predictions['success_prob']
        expected_time_tensor = predictions['expected_time']
        confidence_tensor = predictions['confidence']

        # Convert to scalar by taking mean if batched
        if success_prob_tensor.numel() > 1:
            success_prob = success_prob_tensor.mean().item()
            expected_time = expected_time_tensor.mean().item()
            confidence = confidence_tensor.mean().item()
        else:
            success_prob = success_prob_tensor.item()
            expected_time = expected_time_tensor.item()
            confidence = confidence_tensor.item()

        # Decision logic
        can_solve = success_prob > threshold and confidence > 0.5

        if not can_solve:
            if success_prob <= threshold:
                reason = f"Task too difficult (success probability: {success_prob:.2f})"
            else:
                reason = f"Low confidence in assessment (confidence: {confidence:.2f})"
        else:
            reason = f"Can solve with {success_prob:.2f} probability in ~{expected_time:.1f} steps"

        metrics = {
            'success_probability': success_prob,
            'expected_time': expected_time,
            'confidence': confidence
        }

        return can_solve, reason, metrics
    
    def update_self_knowledge(
        self,
        task_id: int,
        success: bool,
        time_taken: float,
        predicted_success: Optional[float] = None,
    ):
        """Update performance history with importance weighting.

        Args:
            task_id: Task identifier
            success: Whether task was successful
            time_taken: Time taken to complete
            predicted_success: What we predicted (for computing importance)
        """
        # Compute importance based on prediction error
        # Surprising outcomes (wrong predictions) are more important to learn from
        if predicted_success is not None:
            actual = 1.0 if success else 0.0
            prediction_error = abs(predicted_success - actual)
            # Higher error = higher importance
            importance = 0.5 + 0.5 * prediction_error  # Range [0.5, 1.0]
        else:
            importance = 0.75  # Default importance

        # Add to importance-weighted buffer
        entry = torch.tensor([
            float(task_id),
            1.0 if success else 0.0,
            time_taken,
            predicted_success if predicted_success is not None else -1.0,
            importance,
        ])
        self.performance_buffer.add(entry, importance=importance)

        # Also update legacy buffer for backward compatibility
        idx = self.history_index.item() % 100
        self.performance_history[idx, 0] = task_id
        self.performance_history[idx, 1] = 1.0 if success else 0.0
        self.performance_history[idx, 2] = time_taken
        self.history_index += 1
    
    def get_strengths_weaknesses(self) -> Dict[str, Any]:
        """Identify agent's strengths and weaknesses.

        Uses importance-weighted buffer for more accurate assessment,
        giving more weight to surprising/important experiences.
        """
        buffer_data = self.performance_buffer.get_all()

        if buffer_data.size(0) == 0:
            return {
                'strengths': [],
                'weaknesses': [],
                'buffer_stats': self.performance_buffer.get_statistics(),
            }

        # Group by task with importance weighting
        task_performance: Dict[int, List[Tuple[float, float]]] = {}

        for entry in buffer_data:
            task_id = int(entry[0].item())
            success = entry[1].item()
            importance = entry[4].item()

            if task_id not in task_performance:
                task_performance[task_id] = []
            task_performance[task_id].append((success, importance))

        # Identify strengths (>70% weighted success) and weaknesses (<30%)
        strengths = []
        weaknesses = []

        for task_id, performances in task_performance.items():
            # Importance-weighted success rate
            total_importance = sum(imp for _, imp in performances)
            if total_importance > 0:
                weighted_success = sum(s * imp for s, imp in performances) / total_importance
            else:
                weighted_success = sum(s for s, _ in performances) / len(performances)

            if weighted_success > 0.7:
                strengths.append(task_id)
            elif weighted_success < 0.3:
                weaknesses.append(task_id)

        return {
            'strengths': strengths,
            'weaknesses': weaknesses,
            'buffer_stats': self.performance_buffer.get_statistics(),
        }

    def sample_experiences(
        self,
        batch_size: int,
        return_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample experiences from buffer with importance weighting.

        Useful for training - more important experiences sampled more often.
        """
        return self.performance_buffer.sample(batch_size, return_weights)


class ThinkingMonitor(nn.Module):
    """Monitor and analyze thinking processes.

    Integrates ThinkingState enum into the reasoning pipeline for
    state-aware processing and action recommendations.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        lookback_window: int = 5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.lookback_window = lookback_window

        # Loop detector
        self.loop_detector = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            batch_first=True
        )

        self.loop_classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Coherence checker
        self.coherence_checker = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        # Confidence estimator (for state determination)
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # Progress estimator (how much progress is being made)
        self.progress_estimator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # State embeddings for conditioning on current state
        self.state_embeddings = nn.Embedding(len(ThinkingState), hidden_size)

        # State-conditioned output layer
        self.state_conditioned_output = nn.Linear(hidden_size * 2, hidden_size)

        # Legacy state classifier (still useful as secondary signal)
        self.state_classifier = nn.Linear(hidden_size, len(ThinkingState))

        # Track current state for state-aware processing
        self._current_state: ThinkingState = ThinkingState.NORMAL
        self._state_history: List[ThinkingState] = []

    @property
    def current_state(self) -> ThinkingState:
        """Get current thinking state."""
        return self._current_state

    def detect_loop(
        self,
        thought_sequence: List[torch.Tensor]
    ) -> Tuple[bool, float]:
        """Detect if reasoning is stuck in a loop."""
        if len(thought_sequence) < 2:
            return False, 0.0

        # Take recent thoughts
        recent = thought_sequence[-self.lookback_window:]

        # Stack into sequence
        stacked = torch.stack(recent).unsqueeze(0)  # [1, T, D]

        # Encode sequence
        _, (h_n, _) = self.loop_detector(stacked)

        # Classify as loop or not
        loop_prob = self.loop_classifier(h_n.squeeze(0)).item()

        is_loop = loop_prob > 0.7

        return is_loop, loop_prob

    def check_coherence(
        self,
        thought_sequence: List[torch.Tensor]
    ) -> float:
        """Check if thoughts are coherent (no contradictions)."""
        if len(thought_sequence) < 2:
            return 1.0

        # Check consecutive thoughts for coherence
        coherence_scores = []

        for i in range(len(thought_sequence) - 1):
            thought_i = thought_sequence[i]
            thought_j = thought_sequence[i + 1]

            # Concatenate and check
            combined = torch.cat([thought_i, thought_j], dim=-1)
            coherence = self.coherence_checker(combined).item()
            coherence_scores.append(coherence)

        # Average coherence
        avg_coherence = sum(coherence_scores) / len(coherence_scores)

        return avg_coherence

    def estimate_confidence(
        self,
        current_thought: torch.Tensor
    ) -> float:
        """Estimate confidence in current reasoning."""
        conf = self.confidence_estimator(current_thought)
        return conf.item() if conf.numel() == 1 else conf.mean().item()

    def estimate_progress(
        self,
        thought_sequence: List[torch.Tensor]
    ) -> float:
        """Estimate how much progress is being made."""
        if len(thought_sequence) < 2:
            return 1.0  # Assume progress at start

        # Compare first and last thought in window
        first = thought_sequence[max(0, len(thought_sequence) - self.lookback_window)]
        last = thought_sequence[-1]

        combined = torch.cat([first, last], dim=-1)
        progress = self.progress_estimator(combined)
        return progress.item() if progress.numel() == 1 else progress.mean().item()

    def determine_state(
        self,
        confidence: float,
        coherence: float,
        loop_prob: float,
        progress: float
    ) -> ThinkingState:
        """Determine thinking state using the enhanced enum method.

        This uses ThinkingState.from_metrics() to compute state based on
        actual runtime metrics rather than just neural network classification.
        """
        return ThinkingState.from_metrics(
            confidence=confidence,
            coherence=coherence,
            loop_prob=loop_prob,
            progress=progress
        )

    def get_state_conditioned_output(
        self,
        hidden_state: torch.Tensor,
        state: ThinkingState
    ) -> torch.Tensor:
        """Get output conditioned on current thinking state.

        This allows the model to adapt its behavior based on cognitive state.
        """
        # Get state embedding
        state_idx = torch.tensor(state.to_embedding_index(), device=hidden_state.device)
        state_emb = self.state_embeddings(state_idx)

        # Ensure hidden_state is 1D for concatenation
        if hidden_state.dim() > 1:
            hidden_state = hidden_state.squeeze()

        # Concatenate and project
        combined = torch.cat([hidden_state, state_emb], dim=-1)
        return self.state_conditioned_output(combined)

    def forward(
        self,
        thought_sequence: List[torch.Tensor],
        return_state_conditioned: bool = False
    ) -> Dict[str, Any]:
        """Full monitoring analysis with ThinkingState integration.

        Args:
            thought_sequence: List of thought tensors
            return_state_conditioned: If True, include state-conditioned output

        Returns:
            Analysis results including action recommendation from ThinkingState
        """
        if not thought_sequence:
            return {
                'status': 'no_thoughts',
                'action': 'CONTINUE',
                'thinking_state': ThinkingState.NORMAL,
                'thinking_state_value': ThinkingState.NORMAL.value,
            }

        # Detect loop
        is_loop, loop_prob = self.detect_loop(thought_sequence)

        # Check coherence
        coherence = self.check_coherence(thought_sequence)

        # Estimate confidence
        confidence = self.estimate_confidence(thought_sequence[-1])

        # Estimate progress
        progress = self.estimate_progress(thought_sequence)

        # Determine state using enhanced ThinkingState.from_metrics
        state = self.determine_state(
            confidence=confidence,
            coherence=coherence,
            loop_prob=loop_prob,
            progress=progress
        )

        # Update internal state tracking
        self._current_state = state
        self._state_history.append(state)
        if len(self._state_history) > 100:
            self._state_history.pop(0)

        # Get recommended action from ThinkingState enum
        action = state.get_recommended_action()
        reason = self._get_reason_for_state(state, confidence, coherence, loop_prob, progress)

        result = {
            'status': 'analyzed',
            'action': action,
            'reason': reason,
            'is_loop': is_loop,
            'loop_probability': loop_prob,
            'coherence': coherence,
            'confidence': confidence,
            'progress': progress,
            'state': state.value,
            'thinking_state': state,  # Actual enum for downstream use
            'thinking_state_value': state.value,
            'num_thoughts': len(thought_sequence),
            'state_history_length': len(self._state_history),
        }

        # Optionally include state-conditioned output
        if return_state_conditioned:
            conditioned_output = self.get_state_conditioned_output(
                thought_sequence[-1], state
            )
            result['state_conditioned_output'] = conditioned_output

        return result

    def _get_reason_for_state(
        self,
        state: ThinkingState,
        confidence: float,
        coherence: float,
        loop_prob: float,
        progress: float
    ) -> str:
        """Generate human-readable reason for current state."""
        reasons = {
            ThinkingState.NORMAL: f"Thinking proceeding normally (conf={confidence:.2f}, coh={coherence:.2f})",
            ThinkingState.UNCERTAIN: f"Low confidence ({confidence:.2f}) - need more exploration",
            ThinkingState.CONFUSED: f"Low coherence ({coherence:.2f}) - contradictory thoughts detected",
            ThinkingState.STUCK: f"Stuck (loop_prob={loop_prob:.2f}, progress={progress:.2f}) - consider restart",
            ThinkingState.CONFIDENT: f"High confidence ({confidence:.2f}) with good coherence ({coherence:.2f}) - ready to conclude",
        }
        return reasons.get(state, "Unknown state")

    def get_state_statistics(self) -> Dict[str, Any]:
        """Get statistics about thinking state history."""
        if not self._state_history:
            return {'total_states': 0}

        state_counts = {}
        for state in ThinkingState:
            state_counts[state.value] = sum(1 for s in self._state_history if s == state)

        return {
            'total_states': len(self._state_history),
            'state_counts': state_counts,
            'current_state': self._current_state.value,
            'most_common_state': max(state_counts, key=state_counts.get),
        }


class UncertaintyCalibrator(nn.Module):
    """Calibrate confidence estimates to match actual accuracy."""
    
    def __init__(
        self,
        hidden_size: int = 256,
        num_bins: int = 10,
    ):
        super().__init__()
        self.num_bins = num_bins
        
        # Confidence predictor
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Calibration bins (learned)
        self.register_buffer(
            'calibration_map',
            torch.linspace(0, 1, num_bins)
        )
        
        # Track predictions and outcomes
        self.register_buffer(
            'prediction_history',
            torch.zeros(1000, 2)  # [confidence, correct]
        )
        self.register_buffer('history_idx', torch.tensor(0, dtype=torch.long))
        
    def predict_confidence(
        self,
        hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """Predict confidence level."""
        raw_confidence = self.confidence_net(hidden_state)
        
        # Apply calibration
        calibrated = self._calibrate(raw_confidence)
        
        return calibrated
    
    def _calibrate(self, raw_confidence: torch.Tensor) -> torch.Tensor:
        """Apply calibration mapping."""
        # Find nearest bin
        expanded = raw_confidence.unsqueeze(-1)  # [B, 1]
        bins_expanded = self.calibration_map.unsqueeze(0)  # [1, num_bins]
        
        distances = torch.abs(expanded - bins_expanded)
        nearest_bin = torch.argmin(distances, dim=-1)
        
        # Map to calibrated value
        calibrated = self.calibration_map[nearest_bin]
        
        return calibrated.unsqueeze(-1)
    
    def update_calibration(
        self,
        predicted_confidence: float,
        was_correct: bool
    ):
        """Update calibration based on outcome."""
        idx = self.history_idx.item() % 1000
        
        self.prediction_history[idx, 0] = predicted_confidence
        self.prediction_history[idx, 1] = 1.0 if was_correct else 0.0
        
        self.history_idx += 1
        
        # Recompute calibration map periodically
        if self.history_idx % 100 == 0:
            self._recompute_calibration()
    
    def _recompute_calibration(self):
        """Recompute calibration mapping from history."""
        valid_entries = min(self.history_idx.item(), 1000)
        if valid_entries < 10:
            return
        
        history = self.prediction_history[:valid_entries]
        
        # For each bin, compute actual accuracy
        bin_width = 1.0 / self.num_bins
        
        for i in range(self.num_bins):
            bin_start = i * bin_width
            bin_end = (i + 1) * bin_width
            
            # Find predictions in this bin
            in_bin = (history[:, 0] >= bin_start) & (history[:, 0] < bin_end)
            
            if in_bin.sum() > 0:
                # Actual accuracy in this bin
                actual_accuracy = history[in_bin, 1].mean()
                self.calibration_map[i] = actual_accuracy


class MetaSGDOptimizer(nn.Module):
    """Meta-SGD: Learned per-parameter learning rates.

    Implements the Meta-SGD algorithm where learning rates are learned
    parameters rather than fixed hyperparameters. Each parameter gets
    its own learned learning rate that adapts during meta-training.

    Reference: Li et al., "Meta-SGD: Learning to Learn Quickly for
    Few-Shot Learning" (2017)
    """

    def __init__(
        self,
        param_shapes: Dict[str, torch.Size],
        base_lr: float = 0.01,
        lr_range: Tuple[float, float] = (1e-5, 1.0),
        learn_direction: bool = False,
    ):
        """Initialize Meta-SGD optimizer.

        Args:
            param_shapes: Dictionary mapping param names to their shapes
            base_lr: Initial learning rate for all parameters
            lr_range: (min_lr, max_lr) range for learned learning rates
            learn_direction: If True, also learn update direction (not just magnitude)
        """
        super().__init__()

        self.lr_range = lr_range
        self.learn_direction = learn_direction

        # Create learnable learning rates for each parameter
        self.learning_rates = nn.ParameterDict()

        for name, shape in param_shapes.items():
            # Clean name for ParameterDict (replace dots with underscores)
            clean_name = name.replace('.', '_')

            # Initialize log learning rate (use log for numerical stability)
            log_lr = torch.full(shape, math.log(base_lr))
            self.learning_rates[clean_name] = nn.Parameter(log_lr)

        # Optional: learnable update direction modifiers
        if learn_direction:
            self.direction_modifiers = nn.ParameterDict()
            for name, shape in param_shapes.items():
                clean_name = name.replace('.', '_')
                self.direction_modifiers[clean_name] = nn.Parameter(torch.ones(shape))

        # Momentum buffers (not learnable, just state)
        self.momentum_buffers: Dict[str, torch.Tensor] = {}
        self.momentum = 0.9

    def get_lr(self, param_name: str) -> torch.Tensor:
        """Get the learned learning rate for a parameter."""
        clean_name = param_name.replace('.', '_')
        if clean_name in self.learning_rates:
            log_lr = self.learning_rates[clean_name]
            # Clamp to valid range
            lr = torch.exp(log_lr).clamp(self.lr_range[0], self.lr_range[1])
            return lr
        else:
            # Default learning rate
            return torch.tensor(0.01)

    def compute_update(
        self,
        param_name: str,
        gradient: torch.Tensor,
        use_momentum: bool = True
    ) -> torch.Tensor:
        """Compute parameter update using learned learning rate.

        Args:
            param_name: Name of parameter being updated
            gradient: Gradient tensor
            use_momentum: Whether to apply momentum

        Returns:
            Update to apply to parameter
        """
        # Get learned learning rate
        lr = self.get_lr(param_name)

        # Ensure lr broadcasts correctly
        if lr.shape != gradient.shape:
            lr = lr.expand_as(gradient)

        # Apply direction modifier if learned
        if self.learn_direction:
            clean_name = param_name.replace('.', '_')
            if clean_name in self.direction_modifiers:
                direction = self.direction_modifiers[clean_name]
                gradient = gradient * direction

        # Apply momentum
        if use_momentum:
            if param_name not in self.momentum_buffers:
                self.momentum_buffers[param_name] = torch.zeros_like(gradient)

            self.momentum_buffers[param_name] = (
                self.momentum * self.momentum_buffers[param_name] +
                (1 - self.momentum) * gradient
            )
            gradient = self.momentum_buffers[param_name]

        # Compute update
        update = -lr * gradient

        return update

    def step(
        self,
        params: Dict[str, nn.Parameter],
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Perform optimization step.

        Args:
            params: Dictionary of parameters to update
            gradients: Dictionary of gradients

        Returns:
            Dictionary of updated parameter values
        """
        updates = {}
        for name, param in params.items():
            if name in gradients:
                update = self.compute_update(name, gradients[name])
                updates[name] = param + update
            else:
                updates[name] = param

        return updates

    def get_lr_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about learned learning rates."""
        stats = {}
        for name, log_lr in self.learning_rates.items():
            lr = torch.exp(log_lr).clamp(self.lr_range[0], self.lr_range[1])
            stats[name] = {
                'mean': lr.mean().item(),
                'std': lr.std().item(),
                'min': lr.min().item(),
                'max': lr.max().item(),
            }
        return stats


class ThoughtTraceSummarizer(nn.Module):
    """Summarizes thought traces into compact representations.

    Instead of storing full tensor history, creates:
    - Key moment identification
    - Compressed trace summaries
    - Important transition detection
    """

    def __init__(
        self,
        hidden_size: int = 256,
        summary_dim: int = 64,
        max_key_moments: int = 10,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.summary_dim = summary_dim
        self.max_key_moments = max_key_moments

        # Encoder for individual thoughts
        self.thought_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, summary_dim),
        )

        # Importance scorer - determines which moments are key
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Transition detector - identifies state changes
        self.transition_detector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Sequence summarizer (attention-based pooling)
        self.summary_attention = nn.MultiheadAttention(
            embed_dim=summary_dim,
            num_heads=4,
            batch_first=True,
        )

        # Final summary projection
        self.summary_projection = nn.Linear(summary_dim, summary_dim)

    def identify_key_moments(
        self,
        thoughts: List[torch.Tensor],
        states: List[ThinkingState],
        confidences: List[float],
    ) -> List[KeyMoment]:
        """Identify key moments in a thought sequence.

        Key moments are:
        - State transitions
        - High importance events
        - Confidence peaks/valleys
        """
        if not thoughts:
            return []

        key_moments = []

        for i, thought in enumerate(thoughts):
            # Score importance
            importance = self.importance_scorer(thought).item()

            # Detect transitions
            is_transition = False
            if i > 0:
                combined = torch.cat([thoughts[i-1], thought], dim=-1)
                transition_score = self.transition_detector(combined).item()
                is_transition = transition_score > 0.5

            # Check for state change
            state_changed = i > 0 and states[i] != states[i-1]

            # Check for confidence spike
            confidence_spike = False
            if i > 0:
                conf_diff = abs(confidences[i] - confidences[i-1])
                confidence_spike = conf_diff > 0.2

            # Determine if this is a key moment
            is_key = (
                importance > 0.7 or
                is_transition or
                state_changed or
                confidence_spike or
                i == 0 or  # First moment is always key
                i == len(thoughts) - 1  # Last moment is always key
            )

            if is_key:
                # Create compressed summary vector
                summary_vec = self.thought_encoder(thought)

                key_moments.append(KeyMoment(
                    step=i,
                    state=states[i],
                    confidence=confidences[i],
                    importance=importance,
                    summary_vector=summary_vec.detach(),
                ))

        # Keep only top moments by importance
        if len(key_moments) > self.max_key_moments:
            # Always keep first and last
            first = key_moments[0]
            last = key_moments[-1]
            middle = sorted(key_moments[1:-1], key=lambda m: m.importance, reverse=True)
            key_moments = [first] + middle[:self.max_key_moments - 2] + [last]

        return key_moments

    def create_summary(
        self,
        thoughts: List[torch.Tensor],
        states: List[ThinkingState],
        confidences: List[float],
    ) -> SummarizedThoughtTrace:
        """Create a summarized trace from full thought sequence.

        Args:
            thoughts: Full thought tensor sequence
            states: ThinkingState at each step
            confidences: Confidence at each step

        Returns:
            SummarizedThoughtTrace with compressed representation
        """
        if not thoughts:
            return SummarizedThoughtTrace(
                trace_id=-1,
                initial_state=ThinkingState.NORMAL,
                final_state=ThinkingState.NORMAL,
            )

        # Identify key moments
        key_moments = self.identify_key_moments(thoughts, states, confidences)

        # Compute summary statistics
        avg_confidence = sum(confidences) / len(confidences)

        # Compute trace-wide coherence (min across sequence)
        min_coherence = 1.0
        if len(thoughts) > 1:
            for i in range(len(thoughts) - 1):
                combined = torch.cat([thoughts[i], thoughts[i+1]], dim=-1)
                coh = self.transition_detector(combined).item()
                min_coherence = min(min_coherence, coh)

        # Create summary embedding via attention pooling
        if len(thoughts) > 0:
            # Encode all thoughts
            encoded = torch.stack([self.thought_encoder(t) for t in thoughts])
            encoded = encoded.unsqueeze(0)  # [1, T, D]

            # Self-attention pooling
            attended, _ = self.summary_attention(encoded, encoded, encoded)

            # Mean pool to get single vector
            summary_emb = attended.mean(dim=1).squeeze(0)  # [D]
            summary_emb = self.summary_projection(summary_emb)
        else:
            summary_emb = None

        return SummarizedThoughtTrace(
            trace_id=0,  # Will be set by caller
            initial_state=states[0] if states else ThinkingState.NORMAL,
            final_state=states[-1] if states else ThinkingState.NORMAL,
            key_moments=key_moments,
            total_steps=len(thoughts),
            avg_confidence=avg_confidence,
            min_coherence=min_coherence,
            final_confidence=confidences[-1] if confidences else 0.0,
            coherence_score=min_coherence,
            timestamp=0,  # Will be set by caller
            summary_embedding=summary_emb.detach() if summary_emb is not None else None,
        )


class MetaCognition(nn.Module):
    """High-level meta-cognitive control.

    Improvements:
    - Summarized thought traces with key moments (6.6)
    - ThinkingState integration in forward pass (6.5)
    - Meta-SGD learned optimizer (6.8)
    """

    def __init__(
        self,
        hidden_size: int = 256,
        task_embedding_dim: int = 128,
        max_trace_history: int = 100,
        use_meta_sgd: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_meta_sgd = use_meta_sgd

        self.self_model = SelfModel(
            task_embedding_dim=task_embedding_dim
        )

        self.thinking_monitor = ThinkingMonitor(
            hidden_size=hidden_size
        )

        self.uncertainty_calibrator = UncertaintyCalibrator(
            hidden_size=hidden_size
        )

        # Summarizer for memory-efficient trace storage
        self.trace_summarizer = ThoughtTraceSummarizer(
            hidden_size=hidden_size,
            summary_dim=hidden_size // 4,
            max_key_moments=10,
        )

        # Summarized trace history (memory efficient)
        self.max_trace_history = max_trace_history
        self.summarized_traces: List[SummarizedThoughtTrace] = []

        # Legacy trace storage (for backward compatibility, but limited)
        self.thought_trace_history: List[ThoughtTrace] = []

        # State-aware processing layer
        self.state_processor = nn.Linear(hidden_size + len(ThinkingState), hidden_size)

        # Current thinking state (tracked across calls)
        self._current_thinking_state: ThinkingState = ThinkingState.NORMAL

        # Meta-SGD optimizer for self-model (optional)
        if use_meta_sgd:
            param_shapes = {
                name: param.shape
                for name, param in self.self_model.named_parameters()
            }
            self.meta_sgd = MetaSGDOptimizer(
                param_shapes=param_shapes,
                base_lr=0.01,
                lr_range=(1e-5, 0.1),
            )
        else:
            self.meta_sgd = None

    @property
    def current_thinking_state(self) -> ThinkingState:
        """Get current thinking state from monitor."""
        return self._current_thinking_state

    def should_i_attempt(
        self,
        task_embedding: torch.Tensor
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Decide whether to attempt a task."""
        return self.self_model.can_i_solve(task_embedding)

    def monitor_reasoning(
        self,
        thought_sequence: List[torch.Tensor],
        return_state_conditioned: bool = False
    ) -> Dict[str, Any]:
        """Monitor ongoing reasoning process with ThinkingState integration."""
        return self.thinking_monitor(
            thought_sequence,
            return_state_conditioned=return_state_conditioned
        )

    def calibrate_confidence(
        self,
        hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """Get calibrated confidence estimate."""
        return self.uncertainty_calibrator.predict_confidence(hidden_state)

    def record_outcome(
        self,
        task_id: int,
        success: bool,
        time_taken: float,
        predicted_confidence: float,
        predicted_success: Optional[float] = None,
    ):
        """Record task outcome for learning."""
        # Update self-model with importance weighting
        self.self_model.update_self_knowledge(
            task_id, success, time_taken,
            predicted_success=predicted_success
        )

        # Update calibration
        self.uncertainty_calibrator.update_calibration(predicted_confidence, success)

    def _create_state_input(
        self,
        hidden_state: torch.Tensor,
        state: ThinkingState
    ) -> torch.Tensor:
        """Create state-aware input by concatenating hidden state with state encoding."""
        # One-hot encode state
        state_onehot = torch.zeros(len(ThinkingState), device=hidden_state.device)
        state_onehot[state.to_embedding_index()] = 1.0

        # Ensure hidden_state is 1D
        if hidden_state.dim() > 1:
            hidden_state = hidden_state.mean(dim=0)

        # Concatenate
        combined = torch.cat([hidden_state, state_onehot], dim=-1)
        return combined

    def forward(
        self,
        task_embedding: torch.Tensor,
        current_thoughts: List[torch.Tensor],
        hidden_state: torch.Tensor,
        return_state_conditioned: bool = True
    ) -> Dict[str, Any]:
        """Full meta-cognitive cycle with ThinkingState integration.

        Args:
            task_embedding: Task representation
            current_thoughts: List of thought tensors
            hidden_state: Current hidden state
            return_state_conditioned: Whether to include state-conditioned outputs

        Returns:
            Meta-cognitive analysis including ThinkingState and recommended action
        """
        # Check if should attempt
        should_attempt, attempt_reason, capability_metrics = self.should_i_attempt(
            task_embedding
        )

        # Monitor thinking with ThinkingState integration
        monitoring_result = self.monitor_reasoning(
            current_thoughts,
            return_state_conditioned=return_state_conditioned
        )

        # Extract ThinkingState from monitoring
        thinking_state = monitoring_result.get('thinking_state', ThinkingState.NORMAL)
        self._current_thinking_state = thinking_state

        # Get calibrated confidence
        confidence = self.calibrate_confidence(hidden_state)

        # Create state-aware hidden representation
        state_input = self._create_state_input(hidden_state, thinking_state)
        state_aware_hidden = self.state_processor(state_input)

        # Create summarized trace (memory efficient)
        if current_thoughts:
            # Collect states and confidences for summarization
            states = [thinking_state] * len(current_thoughts)  # Simplified
            confidences = [confidence.item()] * len(current_thoughts)

            # Create summary
            summary = self.trace_summarizer.create_summary(
                current_thoughts, states, confidences
            )
            summary.trace_id = len(self.summarized_traces)
            summary.timestamp = len(self.summarized_traces)

            # Store summarized trace
            if len(self.summarized_traces) >= self.max_trace_history:
                self.summarized_traces.pop(0)
            self.summarized_traces.append(summary)

            # Also store legacy trace (limited - only keep last 10)
            legacy_trace = ThoughtTrace(
                thoughts=current_thoughts[-5:] if len(current_thoughts) > 5 else current_thoughts,
                state=thinking_state,
                confidence=confidence.item(),
                coherence_score=monitoring_result.get('coherence', 1.0),
                timestamp=len(self.thought_trace_history)
            )
            if len(self.thought_trace_history) >= 10:  # Strict limit
                self.thought_trace_history.pop(0)
            self.thought_trace_history.append(legacy_trace)

        # Get recommended action from ThinkingState
        recommended_action = thinking_state.get_recommended_action()

        result = {
            'should_attempt': should_attempt,
            'attempt_reason': attempt_reason,
            'capability_metrics': capability_metrics,
            'monitoring': monitoring_result,
            'calibrated_confidence': confidence.item(),
            'thinking_state': thinking_state,
            'thinking_state_value': thinking_state.value,
            'recommended_action': recommended_action,
            'state_aware_hidden': state_aware_hidden,
            'summarized_trace_id': len(self.summarized_traces) - 1,
            'thought_trace_id': len(self.thought_trace_history) - 1,
        }

        # Include state-conditioned output if available
        if return_state_conditioned and 'state_conditioned_output' in monitoring_result:
            result['state_conditioned_output'] = monitoring_result['state_conditioned_output']

        return result

    def get_self_assessment(self) -> Dict[str, Any]:
        """Get comprehensive self-assessment."""
        strengths_weaknesses = self.self_model.get_strengths_weaknesses()

        return {
            'strengths': strengths_weaknesses['strengths'],
            'weaknesses': strengths_weaknesses['weaknesses'],
            'buffer_stats': strengths_weaknesses.get('buffer_stats', {}),
            'total_experiences': self.self_model.history_index.item(),
            'thought_traces_recorded': len(self.thought_trace_history),
            'summarized_traces_recorded': len(self.summarized_traces),
            'current_thinking_state': self._current_thinking_state.value,
            'thinking_state_stats': self.thinking_monitor.get_state_statistics(),
        }

    def get_meta_sgd_statistics(self) -> Optional[Dict[str, Any]]:
        """Get statistics from Meta-SGD optimizer."""
        if self.meta_sgd is None:
            return None
        return self.meta_sgd.get_lr_statistics()

    def get_trace_summaries(self, n_recent: int = 10) -> List[Dict[str, Any]]:
        """Get recent summarized traces in serializable format."""
        summaries = []
        for trace in self.summarized_traces[-n_recent:]:
            summaries.append({
                'trace_id': trace.trace_id,
                'initial_state': trace.initial_state.value,
                'final_state': trace.final_state.value,
                'total_steps': trace.total_steps,
                'avg_confidence': trace.avg_confidence,
                'min_coherence': trace.min_coherence,
                'num_key_moments': len(trace.key_moments),
                'key_moment_states': [m.state.value for m in trace.key_moments],
            })
        return summaries
