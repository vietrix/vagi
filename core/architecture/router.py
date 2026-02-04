"""
Top-K Router for Mixture of Experts.

This module implements the routing mechanism that selects which experts
process each token. Key considerations:

1. Load Balancing: Prevent "expert collapse" where only a few experts are used
2. Auxiliary Losses: Guide the router toward balanced, efficient routing
3. Differentiability: Allow gradients to flow through the routing decision

Mathematical Background:

Given input x, the router produces:
    scores = softmax(x @ W_router)
    top_k_indices = argtop_k(scores)
    weights = normalize(scores[top_k_indices])

Auxiliary Losses:
    - Load Balancing Loss: Minimize variance in expert usage
    - Router Z-Loss: Prevent router logits from becoming too large
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional


def load_balancing_loss(
    router_probs: Tensor,
    expert_indices: Tensor,
    num_experts: int,
) -> Tensor:
    """
    Compute load balancing auxiliary loss.

    This loss encourages the router to distribute tokens evenly across experts.

    Loss = num_experts * Σ_i (f_i * P_i)

    where:
    - f_i = fraction of tokens routed to expert i
    - P_i = mean probability of routing to expert i

    Args:
        router_probs: [batch * seq, num_experts] routing probabilities
        expert_indices: [batch * seq, num_experts_per_tok] selected experts
        num_experts: Total number of experts

    Returns:
        Scalar load balancing loss
    """
    num_tokens = router_probs.shape[0]

    # f_i: fraction of tokens routed to each expert
    # Count how many times each expert is selected
    one_hot = F.one_hot(expert_indices, num_experts).float()  # [batch*seq, k, num_experts]
    tokens_per_expert = one_hot.sum(dim=(0, 1))  # [num_experts]
    f = tokens_per_expert / (num_tokens * expert_indices.shape[1])  # Normalize

    # P_i: mean probability for each expert
    P = router_probs.mean(dim=0)  # [num_experts]

    # Load balancing loss
    return num_experts * (f * P).sum()


def router_z_loss(router_logits: Tensor) -> Tensor:
    """
    Router Z-Loss to prevent large logits.

    This stabilizes training by penalizing large router logits,
    which can cause numerical instability and overconfident routing.

    Loss = (1/N) * Σ (log(Σ_i exp(z_i)))^2

    Args:
        router_logits: [batch * seq, num_experts] raw router outputs

    Returns:
        Scalar z-loss
    """
    # LogSumExp of router logits
    log_z = torch.logsumexp(router_logits, dim=-1)  # [batch * seq]

    # Square and average
    return (log_z ** 2).mean()


class TopKRouter(nn.Module):
    """
    Top-K Router for Mixture of Experts.

    Routes each token to the top-k experts based on learned routing scores.

    Features:
    - Differentiable top-k selection with normalized weights
    - Load balancing auxiliary loss
    - Router z-loss for stability
    - Optional noise injection for exploration during training
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        aux_loss_coef: float = 0.001,
        z_loss_coef: float = 0.001,
        jitter_noise: float = 0.0,
        norm_topk_prob: bool = True,
    ):
        """
        Initialize the router.

        Args:
            hidden_size: Input dimension
            num_experts: Total number of experts to route to
            num_experts_per_tok: Number of experts to select per token (k)
            aux_loss_coef: Weight for load balancing loss
            z_loss_coef: Weight for router z-loss
            jitter_noise: Noise std for exploration (training only)
            norm_topk_prob: Whether to renormalize top-k probabilities
        """
        super().__init__()

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.aux_loss_coef = aux_loss_coef
        self.z_loss_coef = z_loss_coef
        self.jitter_noise = jitter_noise
        self.norm_topk_prob = norm_topk_prob

        # Router projection: hidden_size -> num_experts
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Route tokens to experts.

        Args:
            hidden_states: [num_tokens, hidden_size]

        Returns:
            expert_weights: [num_tokens, num_experts_per_tok] normalized weights
            expert_indices: [num_tokens, num_experts_per_tok] selected expert indices
            aux_loss: Combined auxiliary loss (if training)
        """
        # Compute router logits
        router_logits = self.gate(hidden_states)  # [num_tokens, num_experts]

        # Add noise during training for exploration
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise

        # Compute routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)  # [num_tokens, num_experts]

        # Select top-k experts
        expert_weights, expert_indices = torch.topk(
            router_probs,
            self.num_experts_per_tok,
            dim=-1,
        )

        # Renormalize top-k probabilities (so they sum to 1)
        if self.norm_topk_prob:
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        # Compute auxiliary losses during training
        aux_loss = None
        if self.training:
            # Load balancing loss
            lb_loss = load_balancing_loss(
                router_probs, expert_indices, self.num_experts
            )

            # Router z-loss
            z_loss = router_z_loss(router_logits)

            # Combined auxiliary loss
            aux_loss = self.aux_loss_coef * lb_loss + self.z_loss_coef * z_loss

        return expert_weights, expert_indices, aux_loss


class ExpertChoiceRouter(nn.Module):
    """
    Expert Choice Router (Alternative to Token Choice).

    Instead of tokens choosing experts, experts choose tokens.
    This naturally enforces load balancing.

    Each expert selects top-c tokens based on affinity:
        affinity = softmax(W_expert @ x.T)
        selected_tokens = argtop_c(affinity)

    Benefits:
    - Perfect load balancing (each expert gets exactly c tokens)
    - No auxiliary loss needed
    - Better scaling properties

    Drawbacks:
    - Requires more complex implementation for efficient batching
    - May not work well with very long sequences
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        capacity_factor: float = 1.25,
    ):
        """
        Initialize expert choice router.

        Args:
            hidden_size: Input dimension
            num_experts: Number of experts
            capacity_factor: Multiplier for expert capacity
        """
        super().__init__()

        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        # Expert affinity projection
        self.expert_embeddings = nn.Parameter(
            torch.randn(num_experts, hidden_size)
        )

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Route using expert choice.

        Args:
            hidden_states: [num_tokens, hidden_size]

        Returns:
            expert_weights: [num_experts, capacity, 1] weights for combining
            token_indices: [num_experts, capacity] which tokens each expert processes
            combine_weights: [num_tokens, num_experts] for combining expert outputs
        """
        num_tokens = hidden_states.shape[0]

        # Compute capacity per expert
        capacity = int(self.capacity_factor * num_tokens / self.num_experts)
        capacity = max(capacity, 1)

        # Compute affinity: [num_experts, num_tokens]
        affinity = torch.matmul(
            F.normalize(self.expert_embeddings, dim=-1),
            F.normalize(hidden_states, dim=-1).T,
        )

        # Expert choice: each expert selects top-c tokens
        expert_weights, token_indices = torch.topk(
            affinity, capacity, dim=-1
        )  # [num_experts, capacity]

        # Normalize weights per expert
        expert_weights = F.softmax(expert_weights, dim=-1)

        # Create combine weights for output aggregation
        # [num_tokens, num_experts]
        combine_weights = torch.zeros(
            num_tokens, self.num_experts,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # Fill in the combine weights
        for expert_idx in range(self.num_experts):
            combine_weights[token_indices[expert_idx], expert_idx] = (
                expert_weights[expert_idx]
            )

        return expert_weights.unsqueeze(-1), token_indices, combine_weights


class SwitchRouter(nn.Module):
    """
    Switch Transformer Router (Top-1 Routing).

    Simplified routing where each token goes to exactly one expert.
    More efficient but may sacrifice some quality.

    Used in Switch Transformer (Google) and Mixtral (Mistral AI).
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        capacity_factor: float = 1.25,
        aux_loss_coef: float = 0.01,
        jitter_noise: float = 0.1,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.aux_loss_coef = aux_loss_coef
        self.jitter_noise = jitter_noise

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Top-1 routing.

        Returns:
            expert_weights: [num_tokens, 1]
            expert_indices: [num_tokens, 1]
            aux_loss: Load balancing loss
        """
        router_logits = self.gate(hidden_states)

        # Add noise during training
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise

        router_probs = F.softmax(router_logits, dim=-1)

        # Top-1 selection
        expert_weights, expert_indices = router_probs.max(dim=-1, keepdim=True)

        # Auxiliary loss
        aux_loss = None
        if self.training:
            aux_loss = self.aux_loss_coef * load_balancing_loss(
                router_probs, expert_indices, self.num_experts
            )

        return expert_weights, expert_indices, aux_loss


class SoftMoERouter(nn.Module):
    """
    Soft MoE Router (Fully Differentiable).

    Instead of hard top-k selection, use soft attention over experts.
    All experts receive weighted contributions from all tokens.

    This is more differentiable but loses sparsity benefits.
    Useful for smaller models or when quality > efficiency.

    Soft-MoE formula:
        slots = softmax(W_dispatch @ x.T) @ x  # Expert slots
        output = softmax(W_combine @ slots.T) @ expert(slots)
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_slots_per_expert: int = 1,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.num_slots_per_expert = num_slots_per_expert
        total_slots = num_experts * num_slots_per_expert

        # Dispatch: tokens -> slots
        self.dispatch = nn.Linear(hidden_size, total_slots, bias=False)

        # Combine: slots -> output positions
        self.combine = nn.Linear(hidden_size, total_slots, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Soft routing to all experts.

        Args:
            hidden_states: [num_tokens, hidden_size]

        Returns:
            dispatch_weights: [total_slots, num_tokens] for input combination
            combine_weights: [num_tokens, total_slots] for output combination
        """
        # Dispatch weights: which tokens contribute to which slots
        dispatch_logits = self.dispatch(hidden_states).T  # [total_slots, num_tokens]
        dispatch_weights = F.softmax(dispatch_logits, dim=-1)

        # Combine weights: how to combine slot outputs back to tokens
        combine_logits = self.combine(hidden_states)  # [num_tokens, total_slots]
        combine_weights = F.softmax(combine_logits, dim=-1)

        return dispatch_weights, combine_weights
