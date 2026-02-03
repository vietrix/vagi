"""Meta-cognition: self-awareness and reasoning about reasoning."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F


class ThinkingState(Enum):
    """States in the thinking process."""
    NORMAL = "normal"
    UNCERTAIN = "uncertain"
    CONFUSED = "confused"
    STUCK = "stuck"
    CONFIDENT = "confident"


@dataclass
class ThoughtTrace:
    """Record of a thought process."""
    thoughts: List[torch.Tensor]
    state: ThinkingState
    confidence: float
    coherence_score: float
    timestamp: int


class SelfModel(nn.Module):
    """Model of the agent's own capabilities and limitations."""
    
    def __init__(
        self,
        task_embedding_dim: int = 128,
        capability_dim: int = 64,
        num_capability_types: int = 20,
    ):
        super().__init__()
        
        # Task encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(task_embedding_dim, capability_dim * 2),
            nn.ReLU(),
            nn.Linear(capability_dim * 2, capability_dim)
        )
        
        # Capability predictor
        self.capability_predictor = nn.ModuleDict({
            'success_prob': nn.Sequential(
                nn.Linear(capability_dim, capability_dim),
                nn.ReLU(),
                nn.Linear(capability_dim, 1),
                nn.Sigmoid()
            ),
            'expected_time': nn.Sequential(
                nn.Linear(capability_dim, capability_dim),
                nn.ReLU(),
                nn.Linear(capability_dim, 1),
                nn.Softplus()
            ),
            'confidence': nn.Sequential(
                nn.Linear(capability_dim, capability_dim),
                nn.ReLU(),
                nn.Linear(capability_dim, 1),
                nn.Sigmoid()
            )
        })
        
        # Capability types classifier
        self.capability_classifier = nn.Linear(capability_dim, num_capability_types)
        
        # Self-knowledge: learned statistics about past performance
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
        
        # Predict capabilities
        success_prob = self.capability_predictor['success_prob'](task_features).item()
        expected_time = self.capability_predictor['expected_time'](task_features).item()
        confidence = self.capability_predictor['confidence'](task_features).item()
        
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
        time_taken: float
    ):
        """Update performance history."""
        idx = self.history_index.item() % 100
        
        self.performance_history[idx, 0] = task_id
        self.performance_history[idx, 1] = 1.0 if success else 0.0
        self.performance_history[idx, 2] = time_taken
        
        self.history_index += 1
    
    def get_strengths_weaknesses(self) -> Dict[str, List[int]]:
        """Identify agent's strengths and weaknesses."""
        if self.history_index == 0:
            return {'strengths': [], 'weaknesses': []}
        
        # Analyze performance history
        valid_entries = min(self.history_index.item(), 100)
        history = self.performance_history[:valid_entries]
        
        # Group by task
        task_performance = {}
        for entry in history:
            task_id = int(entry[0].item())
            success = entry[1].item()
            
            if task_id not in task_performance:
                task_performance[task_id] = []
            task_performance[task_id].append(success)
        
        # Identify strengths (>70% success) and weaknesses (<30% success)
        strengths = []
        weaknesses = []
        
        for task_id, successes in task_performance.items():
            success_rate = sum(successes) / len(successes)
            
            if success_rate > 0.7:
                strengths.append(task_id)
            elif success_rate < 0.3:
                weaknesses.append(task_id)
        
        return {
            'strengths': strengths,
            'weaknesses': weaknesses
        }


class ThinkingMonitor(nn.Module):
    """Monitor and analyze thinking processes."""
    
    def __init__(
        self,
        hidden_size: int = 256,
        lookback_window: int = 5,
    ):
        super().__init__()
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
        
        # State classifier
        self.state_classifier = nn.Linear(hidden_size, len(ThinkingState))
        
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
    
    def classify_state(
        self,
        current_thought: torch.Tensor
    ) -> ThinkingState:
        """Classify current thinking state."""
        logits = self.state_classifier(current_thought)
        state_idx = torch.argmax(logits).item()
        
        states = list(ThinkingState)
        return states[state_idx % len(states)]
    
    def forward(
        self,
        thought_sequence: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Full monitoring analysis."""
        if not thought_sequence:
            return {
                'status': 'no_thoughts',
                'action': 'continue'
            }
        
        # Detect loop
        is_loop, loop_prob = self.detect_loop(thought_sequence)
        
        # Check coherence
        coherence = self.check_coherence(thought_sequence)
        
        # Classify state
        state = self.classify_state(thought_sequence[-1])
        
        # Determine action
        if is_loop:
            action = 'STOP'
            reason = f"Detected reasoning loop (confidence: {loop_prob:.2f})"
        elif coherence < 0.5:
            action = 'REVISE'
            reason = f"Low coherence detected ({coherence:.2f}) - contradictory beliefs"
        elif state == ThinkingState.STUCK:
            action = 'RESTART'
            reason = "Thinking process appears stuck"
        elif state == ThinkingState.CONFIDENT:
            action = 'CONCLUDE'
            reason = "High confidence reached"
        else:
            action = 'CONTINUE'
            reason = "Thinking process proceeding normally"
        
        return {
            'status': 'analyzed',
            'action': action,
            'reason': reason,
            'is_loop': is_loop,
            'loop_probability': loop_prob,
            'coherence': coherence,
            'state': state.value,
            'num_thoughts': len(thought_sequence)
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


class MetaCognition(nn.Module):
    """High-level meta-cognitive control."""

    def __init__(
        self,
        hidden_size: int = 256,
        task_embedding_dim: int = 128,
        max_trace_history: int = 100,  # Prevent memory leak
    ):
        super().__init__()

        self.self_model = SelfModel(
            task_embedding_dim=task_embedding_dim
        )

        self.thinking_monitor = ThinkingMonitor(
            hidden_size=hidden_size
        )

        self.uncertainty_calibrator = UncertaintyCalibrator(
            hidden_size=hidden_size
        )

        # Thought history with bounded size (ring buffer)
        self.max_trace_history = max_trace_history
        self.thought_trace_history: List[ThoughtTrace] = []
        
    def should_i_attempt(
        self,
        task_embedding: torch.Tensor
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Decide whether to attempt a task."""
        return self.self_model.can_i_solve(task_embedding)
    
    def monitor_reasoning(
        self,
        thought_sequence: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Monitor ongoing reasoning process."""
        return self.thinking_monitor(thought_sequence)
    
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
        predicted_confidence: float
    ):
        """Record task outcome for learning."""
        # Update self-model
        self.self_model.update_self_knowledge(task_id, success, time_taken)
        
        # Update calibration
        self.uncertainty_calibrator.update_calibration(predicted_confidence, success)
    
    def forward(
        self,
        task_embedding: torch.Tensor,
        current_thoughts: List[torch.Tensor],
        hidden_state: torch.Tensor
    ) -> Dict[str, Any]:
        """Full meta-cognitive cycle."""
        # Check if should attempt
        should_attempt, attempt_reason, capability_metrics = self.should_i_attempt(
            task_embedding
        )
        
        # Monitor thinking
        monitoring_result = self.monitor_reasoning(current_thoughts)
        
        # Get calibrated confidence
        confidence = self.calibrate_confidence(hidden_state)
        
        # Record thought trace (ring buffer to prevent memory leak)
        if current_thoughts:
            trace = ThoughtTrace(
                thoughts=current_thoughts.copy(),
                state=ThinkingState[monitoring_result.get('state', 'NORMAL').upper()],
                confidence=confidence.item(),
                coherence_score=monitoring_result.get('coherence', 1.0),
                timestamp=len(self.thought_trace_history)
            )
            # Ring buffer: remove oldest if at capacity
            if len(self.thought_trace_history) >= self.max_trace_history:
                self.thought_trace_history.pop(0)
            self.thought_trace_history.append(trace)
        
        return {
            'should_attempt': should_attempt,
            'attempt_reason': attempt_reason,
            'capability_metrics': capability_metrics,
            'monitoring': monitoring_result,
            'calibrated_confidence': confidence.item(),
            'thought_trace_id': len(self.thought_trace_history) - 1
        }
    
    def get_self_assessment(self) -> Dict[str, Any]:
        """Get comprehensive self-assessment."""
        strengths_weaknesses = self.self_model.get_strengths_weaknesses()
        
        return {
            'strengths': strengths_weaknesses['strengths'],
            'weaknesses': strengths_weaknesses['weaknesses'],
            'total_experiences': self.self_model.history_index.item(),
            'thought_traces_recorded': len(self.thought_trace_history)
        }
