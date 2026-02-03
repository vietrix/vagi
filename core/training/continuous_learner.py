"""Continuous learning system for AGI - learn from ongoing interactions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from .experience import ExperienceBuffer, ExperienceRecord, QualityGate
from ..base.memory import RecurrentState


@dataclass
class ContinuousLearningConfig:
    """Configuration for continuous learning."""
    buffer_size: int = 10000
    batch_size: int = 32
    learning_rate: float = 1e-4
    update_frequency: int = 10
    min_buffer_size: int = 100
    self_labeling_threshold: float = 0.7
    experience_replay_ratio: float = 0.5
    curriculum_update_interval: int = 100


class SelfSupervisedLabeler(nn.Module):
    """Automatically generate labels from outcomes."""
    
    def __init__(self, hidden_size: int, num_label_types: int = 10):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Label quality predictor
        self.quality_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Outcome-to-label mapper
        self.label_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_label_types)
        )
        
    def generate_labels(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate labels from state transitions and rewards."""
        # Concatenate state and next_state
        combined = torch.cat([state, next_state], dim=-1)
        
        # Predict label quality
        quality = self.quality_predictor(combined)
        
        # Generate labels
        labels = self.label_generator(next_state)
        
        # Weight labels by quality and reward
        weighted_labels = labels * quality * reward.unsqueeze(-1)
        
        return weighted_labels, quality
    
    def filter_high_quality(
        self,
        labels: torch.Tensor,
        quality: torch.Tensor,
        threshold: float = 0.7
    ) -> torch.Tensor:
        """Filter out low quality labels."""
        mask = quality > threshold
        return labels * mask


class ExperienceReplay:
    """Experience replay with prioritization."""
    
    def __init__(
        self,
        buffer_size: int = 10000,
        alpha: float = 0.6,  # Prioritization exponent
        beta: float = 0.4,   # Importance sampling
    ):
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        
        self.buffer = ExperienceBuffer(max_size=buffer_size)
        self.priorities = []
        
    def add(
        self,
        experience: Dict[str, Any],
        priority: Optional[float] = None
    ) -> bool:
        """Add experience with priority."""
        if self.buffer.add(experience):
            if priority is None:
                # Default priority is maximum
                priority = max(self.priorities) if self.priorities else 1.0
            self.priorities.append(priority)
            
            # Trim if exceeded
            if len(self.priorities) > self.buffer_size:
                self.priorities.pop(0)
            
            return True
        return False
    
    def sample(
        self,
        batch_size: int,
        use_priority: bool = True
    ) -> Tuple[List[ExperienceRecord], torch.Tensor, List[int]]:
        """Sample batch with importance sampling."""
        if len(self.buffer) == 0:
            return [], torch.tensor([]), []
        
        if not use_priority or len(self.priorities) == 0:
            # Uniform sampling
            records = self.buffer.sample(batch_size)
            weights = torch.ones(len(records))
            indices = list(range(len(records)))
            return records, weights, indices
        
        # Prioritized sampling
        priorities = torch.tensor(self.priorities, dtype=torch.float32)
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = torch.multinomial(probs, batch_size, replacement=True).tolist()
        
        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        records = [self.buffer.records[i] for i in indices]
        
        return records, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority


class ContinuousLearner:
    """Main continuous learning system."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[ContinuousLearningConfig] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or ContinuousLearningConfig()
        
        self.experience_replay = ExperienceReplay(
            buffer_size=self.config.buffer_size
        )
        
        if hasattr(model, 'cfg') and hasattr(model.cfg, 'hidden_size'):
            hidden_size = model.cfg.hidden_size
        else:
            hidden_size = 512
            
        self.self_labeler = SelfSupervisedLabeler(hidden_size=hidden_size)
        
        self.step_count = 0
        self.update_count = 0
        
        # Statistics
        self.stats = {
            'total_experiences': 0,
            'total_updates': 0,
            'average_loss': 0.0,
            'curriculum_updates': 0
        }
    
    def observe(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_state: Dict[str, torch.Tensor],
        done: bool,
        info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Observe a single experience."""
        # Package experience
        experience = {
            'state': {k: v.cpu() for k, v in state.items()},
            'action': action.cpu(),
            'reward': reward,
            'next_state': {k: v.cpu() for k, v in next_state.items()},
            'done': done,
            'info': info or {}
        }
        
        # Compute priority based on TD error (if value available)
        priority = abs(reward)  # Simple priority
        if 'value' in state and 'value' in next_state:
            td_error = abs(
                reward + 0.99 * next_state['value'].item() - state['value'].item()
            )
            priority = max(priority, td_error)
        
        # Add to replay buffer
        added = self.experience_replay.add(experience, priority=priority)
        
        if added:
            self.stats['total_experiences'] += 1
            self.step_count += 1
            
            # Trigger update if needed
            if self.step_count >= self.config.update_frequency:
                self.update_from_buffer()
                self.step_count = 0
        
        return added
    
    def update_from_buffer(self) -> Dict[str, float]:
        """Update model from experience replay buffer."""
        if len(self.experience_replay.buffer) < self.config.min_buffer_size:
            return {'status': 'insufficient_data'}
        
        # Sample batch
        records, importance_weights, indices = self.experience_replay.sample(
            self.config.batch_size,
            use_priority=True
        )
        
        if len(records) == 0:
            return {'status': 'no_samples'}
        
        # Convert to tensors
        batch = self._records_to_batch(records)
        
        # Generate self-supervised labels
        labels, label_quality = self._generate_labels(batch)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        outputs = self.model(
            input_ids=batch['input_ids'],
            obs=batch['obs'],
            state=batch.get('state'),
            return_loss=True,
            labels=labels,
            targets=batch.get('targets')
        )
        
        # Compute weighted loss
        if 'loss' in outputs:
            loss = outputs['loss']
            weighted_loss = (loss * importance_weights.to(loss.device).mean()).mean()
            
            # Backward and optimize
            weighted_loss.backward()
            self.optimizer.step()
            
            # Update priorities based on loss
            new_priorities = [loss.item()] * len(indices)
            self.experience_replay.update_priorities(indices, new_priorities)
            
            # Update statistics
            self.stats['total_updates'] += 1
            self.stats['average_loss'] = (
                0.9 * self.stats['average_loss'] + 0.1 * weighted_loss.item()
            )
            
            return {
                'status': 'success',
                'loss': weighted_loss.item(),
                'batch_size': len(records)
            }
        
        return {'status': 'no_loss'}
    
    def _records_to_batch(self, records: List[ExperienceRecord]) -> Dict[str, torch.Tensor]:
        """Convert experience records to training batch."""
        batch = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'next_obs': [],
            'dones': []
        }
        
        for record in records:
            data = record.data
            if 'state' in data and 'obs' in data['state']:
                batch['obs'].append(data['state']['obs'])
            if 'action' in data:
                batch['actions'].append(data['action'])
            batch['rewards'].append(data.get('reward', 0.0))
            if 'next_state' in data and 'obs' in data['next_state']:
                batch['next_obs'].append(data['next_state']['obs'])
            batch['dones'].append(data.get('done', False))
        
        # Stack tensors
        device = next(self.model.parameters()).device
        
        result = {}
        if batch['obs']:
            result['obs'] = torch.stack(batch['obs']).to(device)
        if batch['actions']:
            result['input_ids'] = torch.stack(batch['actions']).long().to(device)
            if result['input_ids'].dim() == 1:
                result['input_ids'] = result['input_ids'].unsqueeze(-1)
        else:
            # Default input_ids
            batch_size = len(records)
            result['input_ids'] = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        
        if batch['rewards']:
            result['rewards'] = torch.tensor(batch['rewards'], dtype=torch.float32, device=device)
        
        return result
    
    def _generate_labels(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Generate self-supervised labels from batch."""
        if 'obs' not in batch or 'next_obs' not in batch:
            return None, None
        
        obs = batch['obs']
        next_obs = batch.get('next_obs', obs)
        rewards = batch.get('rewards', torch.zeros(obs.size(0), device=obs.device))
        
        # Generate labels using self-labeler
        labels, quality = self.self_labeler.generate_labels(
            obs, next_obs, rewards
        )
        
        # Filter high quality labels
        labels = self.self_labeler.filter_high_quality(
            labels, quality, threshold=self.config.self_labeling_threshold
        )
        
        return labels, quality
    
    def update_curriculum(self, task_id: int, performance: float):
        """Update curriculum scheduler based on performance."""
        if hasattr(self.model, 'curriculum_scheduler'):
            self.model.curriculum_scheduler.update_performance(task_id, performance)
            self.stats['curriculum_updates'] += 1
    
    def learn_from_episode(
        self,
        episode_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Learn from complete episode."""
        # Add all experiences to buffer
        added_count = 0
        for step_data in episode_data:
            if 'state' in step_data and 'next_state' in step_data:
                added = self.observe(
                    state=step_data['state'],
                    action=step_data.get('action', torch.tensor([0])),
                    reward=step_data.get('reward', 0.0),
                    next_state=step_data['next_state'],
                    done=step_data.get('done', False),
                    info=step_data.get('info')
                )
                if added:
                    added_count += 1
        
        # Trigger update
        if added_count > 0:
            update_result = self.update_from_buffer()
            return {**update_result, 'experiences_added': added_count}
        
        return {'status': 'no_data', 'experiences_added': 0}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            **self.stats,
            'buffer_size': len(self.experience_replay.buffer),
            'buffer_capacity': self.config.buffer_size
        }
