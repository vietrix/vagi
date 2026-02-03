"""Intrinsic motivation system for curiosity-driven exploration."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class IntrinsicRewardConfig:
    """Configuration for intrinsic rewards."""
    curiosity_weight: float = 1.0
    empowerment_weight: float = 0.5
    novelty_weight: float = 0.3
    prediction_error_scale: float = 1.0


class ForwardDynamicsModel(nn.Module):
    """Predict next state from current state and action."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
    ):
        super().__init__()
        
        self.forward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim)
        )
        
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """Predict next state."""
        combined = torch.cat([state, action], dim=-1)
        predicted_next_state = self.forward_net(combined)
        return predicted_next_state


class InverseDynamicsModel(nn.Module):
    """Predict action from state transition."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
    ):
        super().__init__()
        
        self.inverse_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        
    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """Predict action that caused transition."""
        combined = torch.cat([state, next_state], dim=-1)
        predicted_action = self.inverse_net(combined)
        return predicted_action


class CuriosityModule(nn.Module):
    """ICM (Intrinsic Curiosity Module) for exploration."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feature_dim: int = 128,
        hidden_size: int = 256,
    ):
        super().__init__()
        
        # Feature encoder (learns task-relevant features)
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_dim)
        )
        
        # Forward model
        self.forward_model = ForwardDynamicsModel(
            state_dim=feature_dim,
            action_dim=action_dim,
            hidden_size=hidden_size
        )
        
        # Inverse model
        self.inverse_model = InverseDynamicsModel(
            state_dim=feature_dim,
            action_dim=action_dim,
            hidden_size=hidden_size
        )
        
        self.feature_dim = feature_dim
        
    def compute_intrinsic_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """Compute curiosity-based intrinsic reward.
        
        Reward is based on prediction error of forward model.
        """
        # Encode states to features
        state_feat = self.feature_encoder(state)
        next_state_feat = self.feature_encoder(next_state)
        
        # Predict next state features
        predicted_next_feat = self.forward_model(state_feat, action)
        
        # Prediction error = intrinsic reward (surprise)
        prediction_error = F.mse_loss(
            predicted_next_feat,
            next_state_feat,
            reduction='none'
        ).mean(dim=-1)
        
        return prediction_error
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass with both models."""
        # Encode features
        state_feat = self.feature_encoder(state)
        next_state_feat = self.feature_encoder(next_state)
        
        # Forward model prediction
        predicted_next_feat = self.forward_model(state_feat, action)
        forward_loss = F.mse_loss(predicted_next_feat, next_state_feat.detach())
        
        # Inverse model prediction
        predicted_action = self.inverse_model(state_feat, next_state_feat)
        inverse_loss = F.mse_loss(predicted_action, action)
        
        # Intrinsic reward
        intrinsic_reward = self.compute_intrinsic_reward(state, action, next_state)
        
        return {
            'forward_loss': forward_loss,
            'inverse_loss': inverse_loss,
            'intrinsic_reward': intrinsic_reward,
            'state_features': state_feat,
            'next_state_features': next_state_feat
        }


class NoveltyDetector(nn.Module):
    """Detect novel states using episodic memory."""
    
    def __init__(
        self,
        state_dim: int,
        capacity: int = 1000,
        k_nearest: int = 10,
    ):
        super().__init__()
        self.capacity = capacity
        self.k_nearest = k_nearest
        
        # Episodic memory buffer
        self.register_buffer(
            'memory',
            torch.zeros(capacity, state_dim)
        )
        self.register_buffer(
            'memory_count',
            torch.tensor(0, dtype=torch.long)
        )
        
        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim)
        )
        
    def compute_novelty(self, state: torch.Tensor) -> torch.Tensor:
        """Compute novelty based on distance to k-nearest neighbors."""
        # Encode state
        state_encoded = self.encoder(state)
        
        # Get valid memory entries
        valid_count = min(self.memory_count.item(), self.capacity)
        
        if valid_count == 0:
            # Everything is novel at start
            return torch.ones(state.size(0), device=state.device)
        
        # Compute distances to all memory entries
        memory_valid = self.memory[:valid_count]
        
        # Expand for broadcasting
        state_expanded = state_encoded.unsqueeze(1)  # [B, 1, D]
        memory_expanded = memory_valid.unsqueeze(0)  # [1, M, D]
        
        # L2 distances
        distances = torch.norm(
            state_expanded - memory_expanded,
            dim=-1
        )  # [B, M]
        
        # Get k-nearest distances
        k = min(self.k_nearest, valid_count)
        knn_distances, _ = torch.topk(distances, k, dim=-1, largest=False)
        
        # Novelty = mean distance to k-nearest neighbors
        novelty = knn_distances.mean(dim=-1)
        
        return novelty
    
    def add_to_memory(self, state: torch.Tensor):
        """Add state to episodic memory."""
        with torch.no_grad():
            state_encoded = self.encoder(state)

            # Handle both batched and single states
            if state_encoded.dim() == 1:
                state_encoded = state_encoded.unsqueeze(0)

            for s in state_encoded:
                # Flatten if needed to match memory shape
                s_flat = s.flatten()[:self.memory.size(-1)]
                # Pad if too short
                if s_flat.size(0) < self.memory.size(-1):
                    s_flat = F.pad(s_flat, (0, self.memory.size(-1) - s_flat.size(0)))

                idx = self.memory_count.item() % self.capacity
                self.memory[idx] = s_flat
                self.memory_count += 1
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute novelty and update memory."""
        novelty = self.compute_novelty(state)
        self.add_to_memory(state)
        return novelty


class EmpowermentEstimator(nn.Module):
    """Estimate empowerment (mutual information between actions and states)."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        num_samples: int = 10,
    ):
        super().__init__()
        self.num_samples = num_samples
        
        # Action proposal network
        self.action_proposal = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim * 2)  # Mean and log_std
        )
        
        # State prediction network
        self.state_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate empowerment of current state."""
        batch_size = state.size(0)
        
        # Propose action distribution
        action_params = self.action_proposal(state)
        action_mean = action_params[:, :action_params.size(-1)//2]
        action_log_std = action_params[:, action_params.size(-1)//2:]
        action_std = torch.exp(action_log_std)
        
        # Sample multiple actions
        empowerment_scores = []
        
        for _ in range(self.num_samples):
            # Sample action
            epsilon = torch.randn_like(action_mean)
            action = action_mean + action_std * epsilon
            
            # Predict resulting state
            predicted_state = self.state_predictor(
                torch.cat([state, action], dim=-1)
            )
            
            # Compute distinguishability (how different is the outcome?)
            distinguishability = F.mse_loss(
                predicted_state,
                state,
                reduction='none'
            ).mean(dim=-1)
            
            empowerment_scores.append(distinguishability)
        
        # Empowerment = average distinguishability
        empowerment = torch.stack(empowerment_scores).mean(dim=0)
        
        return empowerment


class GoalGenerator(nn.Module):
    """Generate achievable sub-goals automatically."""
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 64,
        hidden_size: int = 256,
    ):
        super().__init__()
        
        # VAE for goal generation
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim * 2)  # mu and logvar
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim)
        )
        
        # Reachability predictor
        self.reachability = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        self.latent_dim = latent_dim
        
    def encode(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode state to latent distribution."""
        params = self.encoder(state)
        mu = params[:, :self.latent_dim]
        logvar = params[:, self.latent_dim:]
        return mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to goal state."""
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def propose_goals(
        self,
        current_state: torch.Tensor,
        num_goals: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propose achievable goals from current state."""
        # Encode current state
        mu, logvar = self.encode(current_state)
        
        # Sample multiple goals
        goals = []
        reachabilities = []
        
        for _ in range(num_goals):
            # Sample from latent space with some noise
            z = self.reparameterize(mu, logvar)
            
            # Add exploration noise
            z = z + torch.randn_like(z) * 0.1
            
            # Decode to goal state
            goal = self.decode(z)
            
            # Predict reachability
            reachability = self.reachability(
                torch.cat([current_state, goal], dim=-1)
            )
            
            goals.append(goal)
            reachabilities.append(reachability)
        
        goals = torch.stack(goals, dim=1)  # [B, num_goals, state_dim]
        reachabilities = torch.stack(reachabilities, dim=1).squeeze(-1)  # [B, num_goals]
        
        return goals, reachabilities
    
    def filter_achievable(
        self,
        current_state: torch.Tensor,
        goals: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """Filter goals by reachability."""
        batch_size, num_goals, state_dim = goals.size()

        # Expand current state
        current_expanded = current_state.unsqueeze(1).expand(-1, num_goals, -1)

        # Reshape for reachability prediction
        current_flat = current_expanded.reshape(-1, state_dim)
        goals_flat = goals.reshape(-1, state_dim)

        # Predict reachability
        reachability = self.reachability(
            torch.cat([current_flat, goals_flat], dim=-1)
        ).view(batch_size, num_goals)

        # Filter achievable goals
        mask = reachability > threshold

        return mask, reachability

    def forward(
        self,
        current_state: torch.Tensor,
        num_goals: int = 5,
        return_best: bool = True
    ) -> torch.Tensor:
        """Generate exploration goal from current state.

        Args:
            current_state: [B, state_dim] current state tensor
            num_goals: Number of candidate goals to generate
            return_best: If True, return single best goal; else return all candidates

        Returns:
            If return_best: [B, state_dim] best achievable goal
            Else: [B, num_goals, state_dim] all candidate goals
        """
        # Generate candidate goals
        goals, reachabilities = self.propose_goals(current_state, num_goals)

        if not return_best:
            return goals

        # Select best goal (highest reachability)
        best_idx = reachabilities.argmax(dim=-1)  # [B]

        # Gather best goal for each batch element
        batch_size = goals.size(0)
        batch_indices = torch.arange(batch_size, device=goals.device)
        best_goals = goals[batch_indices, best_idx]  # [B, state_dim]

        return best_goals


class IntrinsicMotivationSystem(nn.Module):
    """Unified intrinsic motivation system."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[IntrinsicRewardConfig] = None,
    ):
        super().__init__()
        self.config = config or IntrinsicRewardConfig()
        
        self.curiosity = CuriosityModule(
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        self.novelty = NoveltyDetector(
            state_dim=state_dim
        )
        
        self.empowerment = EmpowermentEstimator(
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        self.goal_generator = GoalGenerator(
            state_dim=state_dim
        )
        
    def compute_intrinsic_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute combined intrinsic reward."""
        # Curiosity reward (prediction error)
        curiosity_reward = self.curiosity.compute_intrinsic_reward(
            state, action, next_state
        )
        
        # Novelty reward
        novelty_reward = self.novelty(next_state)
        
        # Empowerment reward
        empowerment_reward = self.empowerment(state)
        
        # Combined intrinsic reward
        intrinsic_reward = (
            self.config.curiosity_weight * curiosity_reward +
            self.config.novelty_weight * novelty_reward +
            self.config.empowerment_weight * empowerment_reward
        )
        
        return {
            'intrinsic_reward': intrinsic_reward,
            'curiosity': curiosity_reward,
            'novelty': novelty_reward,
            'empowerment': empowerment_reward
        }
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass with all losses."""
        # Curiosity model
        curiosity_outputs = self.curiosity(state, action, next_state)
        
        # Compute all intrinsic rewards
        rewards = self.compute_intrinsic_reward(state, action, next_state)
        
        return {
            **curiosity_outputs,
            **rewards
        }
