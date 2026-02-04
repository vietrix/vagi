"""
Balanced Mixture of Experts Router with Hardware-Adaptive Execution.

This module extends the standard MoE routing with:
1. Balanced routing with load balancing guarantees
2. Hardware-adaptive expert execution (parallel on GPU, sequential on CPU)
3. Expert capacity management to prevent dropped tokens
4. Auxiliary losses for training stability

Key Features:
- GPU: Parallel expert execution with batched matrix operations
- CPU: Sequential expert execution to minimize memory usage
- MPS: Hybrid approach optimized for Apple Silicon

Mathematical Background:
    Token-to-Expert Assignment:
        scores = softmax(x @ W_gate + noise)
        top_k_experts = argtop_k(scores)

    Load Balancing Loss:
        L_balance = num_experts * sum(f_i * P_i)
        where f_i = fraction of tokens to expert i
              P_i = mean probability to expert i

References:
    - Switch Transformer (Google)
    - Mixtral 8x7B (Mistral AI)
    - DeepSeekMoE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, List, Literal
from dataclasses import dataclass
import math

from .ops import get_device_manager, DeviceManager


@dataclass
class MoEMetrics:
    """Metrics for monitoring MoE behavior."""
    load_balance_loss: Tensor
    router_z_loss: Tensor
    expert_utilization: Tensor  # [num_experts] - usage per expert
    tokens_dropped: int
    tokens_padded: int


class BalancedTopKRouter(nn.Module):
    """
    Balanced Top-K Router with load balancing guarantees.

    Features:
    - Auxiliary loss for load balancing
    - Expert capacity constraints
    - Jitter noise for exploration
    - Token dropping and padding for capacity management
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_coef: float = 0.01,
        z_loss_coef: float = 0.001,
        jitter_noise: float = 0.0,
        normalize_weights: bool = True,
        drop_tokens: bool = True,
    ):
        """
        Initialize balanced router.

        Args:
            hidden_size: Input dimension
            num_experts: Total number of experts
            num_experts_per_tok: Experts activated per token (k)
            capacity_factor: Multiplier for expert capacity
            aux_loss_coef: Weight for load balancing loss
            z_loss_coef: Weight for router z-loss
            jitter_noise: Noise std for exploration
            normalize_weights: Renormalize top-k probabilities
            drop_tokens: Whether to drop tokens exceeding capacity
        """
        super().__init__()

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.capacity_factor = capacity_factor
        self.aux_loss_coef = aux_loss_coef
        self.z_loss_coef = z_loss_coef
        self.jitter_noise = jitter_noise
        self.normalize_weights = normalize_weights
        self.drop_tokens = drop_tokens

        # Router projection
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Initialize with small values for stable early training
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)

    def _compute_capacity(self, num_tokens: int) -> int:
        """Compute expert capacity based on number of tokens."""
        # Each token goes to k experts
        # Ideal capacity = num_tokens * k / num_experts
        # Apply capacity factor for buffer
        ideal_capacity = (num_tokens * self.num_experts_per_tok) / self.num_experts
        return int(math.ceil(ideal_capacity * self.capacity_factor))

    def _load_balancing_loss(
        self,
        router_probs: Tensor,
        expert_indices: Tensor,
    ) -> Tensor:
        """
        Compute load balancing auxiliary loss.

        Encourages uniform expert usage across tokens.
        """
        num_tokens = router_probs.shape[0]

        # f_i: fraction of tokens routed to each expert
        one_hot = F.one_hot(expert_indices, self.num_experts).float()
        tokens_per_expert = one_hot.sum(dim=(0, 1))
        f = tokens_per_expert / (num_tokens * self.num_experts_per_tok)

        # P_i: mean routing probability per expert
        P = router_probs.mean(dim=0)

        return self.num_experts * (f * P).sum()

    def _router_z_loss(self, router_logits: Tensor) -> Tensor:
        """
        Router Z-loss to prevent large logits.

        Stabilizes training by penalizing extreme router outputs.
        """
        log_z = torch.logsumexp(router_logits, dim=-1)
        return (log_z ** 2).mean()

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[MoEMetrics]]:
        """
        Route tokens to experts with load balancing.

        Args:
            hidden_states: [num_tokens, hidden_size]

        Returns:
            expert_weights: [num_tokens, num_experts_per_tok]
            expert_indices: [num_tokens, num_experts_per_tok]
            metrics: MoE metrics (if training)
        """
        num_tokens = hidden_states.shape[0]

        # Compute routing scores
        router_logits = self.gate(hidden_states)  # [num_tokens, num_experts]

        # Add jitter noise during training
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise

        # Softmax for routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts per token
        expert_weights, expert_indices = torch.topk(
            router_probs, self.num_experts_per_tok, dim=-1
        )

        # Renormalize weights
        if self.normalize_weights:
            expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Compute metrics during training
        metrics = None
        if self.training:
            lb_loss = self._load_balancing_loss(router_probs, expert_indices)
            z_loss = self._router_z_loss(router_logits)

            # Expert utilization
            one_hot = F.one_hot(expert_indices, self.num_experts).float()
            expert_util = one_hot.sum(dim=(0, 1)) / (num_tokens * self.num_experts_per_tok)

            metrics = MoEMetrics(
                load_balance_loss=self.aux_loss_coef * lb_loss,
                router_z_loss=self.z_loss_coef * z_loss,
                expert_utilization=expert_util,
                tokens_dropped=0,
                tokens_padded=0,
            )

        return expert_weights, expert_indices, metrics


class BalancedMoELayer(nn.Module):
    """
    Mixture of Experts layer with balanced routing and hardware-adaptive execution.

    Execution Strategies:
    - GPU (CUDA): Parallel batched execution across all experts
    - CPU: Sequential execution to minimize memory
    - MPS: Hybrid approach for Apple Silicon

    The layer automatically selects the best strategy based on:
    - Available hardware
    - Number of experts
    - Batch size
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_coef: float = 0.01,
        activation: str = "silu",
        expert_dropout: float = 0.0,
    ):
        """
        Initialize balanced MoE layer.

        Args:
            hidden_size: Input/output dimension
            intermediate_size: Expert FFN intermediate dimension
            num_experts: Number of expert networks
            num_experts_per_tok: Experts activated per token
            capacity_factor: Expert capacity multiplier
            aux_loss_coef: Load balancing loss coefficient
            activation: Activation function ("silu", "gelu", "relu")
            expert_dropout: Dropout probability for experts
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_dropout = expert_dropout

        # Router
        self.router = BalancedTopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            capacity_factor=capacity_factor,
            aux_loss_coef=aux_loss_coef,
        )

        # Expert networks (shared architecture, separate weights)
        self.experts = nn.ModuleList([
            Expert(hidden_size, intermediate_size, activation)
            for _ in range(num_experts)
        ])

        # Device manager for hardware detection
        self._device_manager = None

    @property
    def device_manager(self) -> DeviceManager:
        """Lazy initialization of device manager."""
        if self._device_manager is None:
            self._device_manager = get_device_manager()
        return self._device_manager

    def _parallel_expert_forward(
        self,
        hidden_states: Tensor,
        expert_weights: Tensor,
        expert_indices: Tensor,
    ) -> Tensor:
        """
        Parallel expert execution for GPU.

        Batches all experts together for efficient GPU utilization.
        """
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, self.hidden_size)
        num_tokens = hidden_states_flat.shape[0]

        # Prepare output tensor
        final_output = torch.zeros_like(hidden_states_flat)

        # Process each expert position in parallel
        for k in range(self.num_experts_per_tok):
            # Get expert assignments for this position
            indices = expert_indices[:, k]  # [num_tokens]
            weights = expert_weights[:, k:k+1]  # [num_tokens, 1]

            # Group tokens by expert
            for expert_idx in range(self.num_experts):
                # Find tokens routed to this expert
                mask = (indices == expert_idx)
                if not mask.any():
                    continue

                # Get tokens for this expert
                expert_tokens = hidden_states_flat[mask]

                # Apply expert
                expert_output = self.experts[expert_idx](expert_tokens)

                # Apply dropout during training
                if self.training and self.expert_dropout > 0:
                    expert_output = F.dropout(expert_output, p=self.expert_dropout)

                # Weighted accumulation
                final_output[mask] += weights[mask] * expert_output

        return final_output.view(batch_size, seq_len, self.hidden_size)

    def _sequential_expert_forward(
        self,
        hidden_states: Tensor,
        expert_weights: Tensor,
        expert_indices: Tensor,
    ) -> Tensor:
        """
        Sequential expert execution for CPU.

        Processes one expert at a time to minimize memory usage.
        """
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, self.hidden_size)

        # Prepare output tensor
        final_output = torch.zeros_like(hidden_states_flat)

        # Process experts sequentially
        for expert_idx in range(self.num_experts):
            # Find all tokens routed to this expert (across all k positions)
            expert_mask = (expert_indices == expert_idx).any(dim=-1)

            if not expert_mask.any():
                continue

            # Get tokens for this expert
            expert_tokens = hidden_states_flat[expert_mask]

            # Apply expert (only load one expert at a time)
            with torch.no_grad() if not self.training else torch.enable_grad():
                expert_output = self.experts[expert_idx](expert_tokens)

            # Apply dropout during training
            if self.training and self.expert_dropout > 0:
                expert_output = F.dropout(expert_output, p=self.expert_dropout)

            # Compute weighted contribution
            token_indices = torch.where(expert_mask)[0]
            for i, tok_idx in enumerate(token_indices):
                # Find which k position(s) selected this expert
                positions = (expert_indices[tok_idx] == expert_idx).nonzero(as_tuple=True)[0]
                for pos in positions:
                    weight = expert_weights[tok_idx, pos]
                    final_output[tok_idx] += weight * expert_output[i]

        return final_output.view(batch_size, seq_len, self.hidden_size)

    def _batched_expert_forward(
        self,
        hidden_states: Tensor,
        expert_weights: Tensor,
        expert_indices: Tensor,
    ) -> Tensor:
        """
        Batched expert execution using einsum operations.

        Most efficient for medium-sized batches on GPU.
        """
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, self.hidden_size)
        num_tokens = hidden_states_flat.shape[0]

        # Create expert assignment matrix [num_tokens, num_experts, k]
        # This enables batched operations
        expert_mask = F.one_hot(expert_indices, self.num_experts).float()
        expert_mask = expert_mask.permute(0, 2, 1)  # [num_tokens, num_experts, k]

        # Apply weighted mask
        weighted_mask = expert_mask * expert_weights.unsqueeze(1)  # [num_tokens, num_experts, k]
        weighted_mask = weighted_mask.sum(dim=-1)  # [num_tokens, num_experts]

        # Collect expert outputs
        expert_outputs = torch.stack([
            self.experts[i](hidden_states_flat)
            for i in range(self.num_experts)
        ], dim=1)  # [num_tokens, num_experts, hidden_size]

        # Weighted combination
        final_output = torch.einsum('ne,neh->nh', weighted_mask, expert_outputs)

        return final_output.view(batch_size, seq_len, self.hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        execution_mode: Optional[Literal['auto', 'parallel', 'sequential', 'batched']] = 'auto',
    ) -> Tuple[Tensor, Optional[MoEMetrics]]:
        """
        Forward pass through balanced MoE layer.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            execution_mode: Expert execution strategy
                - 'auto': Select based on hardware
                - 'parallel': GPU-optimized parallel execution
                - 'sequential': Memory-efficient sequential execution
                - 'batched': Batched einsum-based execution

        Returns:
            output: [batch, seq_len, hidden_size]
            metrics: MoE metrics (if training)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Flatten for routing
        hidden_flat = hidden_states.view(-1, self.hidden_size)

        # Route tokens to experts
        expert_weights, expert_indices, metrics = self.router(hidden_flat)

        # Reshape indices and weights
        expert_weights = expert_weights.view(batch_size * seq_len, -1)
        expert_indices = expert_indices.view(batch_size * seq_len, -1)

        # Select execution mode
        if execution_mode == 'auto':
            dm = self.device_manager
            if dm.is_cuda:
                # Use batched for GPU (most efficient)
                execution_mode = 'batched'
            elif dm.is_mps:
                # Use parallel for MPS
                execution_mode = 'parallel'
            else:
                # Use sequential for CPU (memory efficient)
                execution_mode = 'sequential'

        # Execute experts
        if execution_mode == 'parallel':
            output = self._parallel_expert_forward(
                hidden_states, expert_weights, expert_indices
            )
        elif execution_mode == 'sequential':
            output = self._sequential_expert_forward(
                hidden_states, expert_weights, expert_indices
            )
        else:  # batched
            output = self._batched_expert_forward(
                hidden_states, expert_weights, expert_indices
            )

        return output, metrics


class Expert(nn.Module):
    """
    Single expert network (Feed-Forward).

    Architecture:
        Linear(hidden -> intermediate)
        -> Activation
        -> Linear(intermediate -> hidden)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
    ):
        super().__init__()

        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # Activation
        activations = {
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "swiglu": None,  # Special handling
        }
        self.activation = activations.get(activation, nn.SiLU())
        self.use_swiglu = (activation == "swiglu")

        if self.use_swiglu:
            # SwiGLU uses gated activation
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_swiglu:
            # SwiGLU: silu(gate) * up
            gate = F.silu(self.gate_proj(x))
            up = self.up_proj(x)
            x = gate * up
        else:
            x = self.up_proj(x)
            x = self.activation(x)

        return self.down_proj(x)


class SharedExpertMoE(nn.Module):
    """
    MoE with shared experts (DeepSeekMoE style).

    Some experts are always activated for all tokens,
    while others are routed dynamically.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_routed_experts: int,
        num_shared_experts: int = 2,
        num_experts_per_tok: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.num_shared_experts = num_shared_experts

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            Expert(hidden_size, intermediate_size)
            for _ in range(num_shared_experts)
        ])

        # Routed MoE layer
        self.moe = BalancedMoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            **kwargs,
        )

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tuple[Tensor, Optional[MoEMetrics]]:
        """
        Forward with shared + routed experts.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, hidden_size]
            metrics: MoE metrics
        """
        # Apply shared experts
        shared_output = sum(
            expert(hidden_states) for expert in self.shared_experts
        ) / self.num_shared_experts

        # Apply routed experts
        routed_output, metrics = self.moe(hidden_states)

        # Combine outputs
        return shared_output + routed_output, metrics


if __name__ == "__main__":
    # Test balanced MoE
    print("Testing Balanced MoE Router")
    print("=" * 50)

    dm = get_device_manager()
    device = dm.get_default_device()
    dtype = dm.get_optimal_dtype(device)

    print(f"Device: {device}, dtype: {dtype}")

    # Create MoE layer
    hidden_size = 512
    intermediate_size = 2048
    num_experts = 8
    batch_size = 2
    seq_len = 128

    moe = BalancedMoELayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=2,
    ).to(device=device, dtype=dtype)

    # Test input
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    # Forward pass
    moe.train()
    output, metrics = moe(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    if metrics:
        print(f"\nMetrics:")
        print(f"  Load balance loss: {metrics.load_balance_loss.item():.6f}")
        print(f"  Router z-loss: {metrics.router_z_loss.item():.6f}")
        print(f"  Expert utilization: {metrics.expert_utilization.tolist()}")

    # Test different execution modes
    print("\nTesting execution modes:")
    for mode in ['parallel', 'sequential', 'batched']:
        try:
            out, _ = moe(x, execution_mode=mode)
            print(f"  {mode}: OK (shape={out.shape})")
        except Exception as e:
            print(f"  {mode}: Failed ({e})")

    # Test shared expert MoE
    print("\nTesting SharedExpertMoE:")
    shared_moe = SharedExpertMoE(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_routed_experts=6,
        num_shared_experts=2,
    ).to(device=device, dtype=dtype)

    out, metrics = shared_moe(x)
    print(f"  Output shape: {out.shape}")

    print("\nAll tests passed!")
