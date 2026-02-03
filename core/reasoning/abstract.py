"""Abstract reasoning and relational thinking modules.

This module provides:
- Relational reasoning with Graph Attention Networks (5.7)
- Causal inference with temporal ordering and Granger causality (5.9)
- Analogy matching with explicit dimension checking (5.8)
- Abstract reasoning with explicit causal/analogical modes (5.6)
- Counterfactual reasoning with plausibility scoring (5.10)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, Literal
from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F

# Configure module logger
logger = logging.getLogger(__name__)


# ============================================================================
# 5.6 Reasoning Modes
# ============================================================================

class ReasoningMode(Enum):
    """Explicit reasoning modes for AbstractReasoner (5.6)."""
    AUTO = "auto"           # Automatically route based on input
    RELATIONAL = "relational"  # Object-relation reasoning
    CAUSAL = "causal"       # Causal inference and intervention
    ANALOGICAL = "analogical"  # Analogy-based reasoning
    COUNTERFACTUAL = "counterfactual"  # What-if reasoning


# ============================================================================
# 5.7 Type Constraints for Graph Attention
# ============================================================================

class NodeType(Enum):
    """Node types for typed graph attention (5.7)."""
    ENTITY = "entity"
    ATTRIBUTE = "attribute"
    ACTION = "action"
    STATE = "state"
    RELATION = "relation"


class EdgeType(Enum):
    """Edge types for typed graph attention (5.7)."""
    HAS_ATTRIBUTE = "has_attribute"
    CAUSES = "causes"
    PART_OF = "part_of"
    TEMPORAL_BEFORE = "temporal_before"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"


class TypedGraphAttention(nn.Module):
    """Graph Attention layer with type constraints (5.7).

    Implements GAT with typed nodes and edges, allowing different
    attention patterns for different relation types.

    Attributes:
        hidden_dim: Hidden dimension for attention
        num_heads: Number of attention heads
        num_node_types: Number of distinct node types
        num_edge_types: Number of distinct edge types
        dropout: Dropout rate for attention weights
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_node_types: int = 5,
        num_edge_types: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Node type embeddings
        self.node_type_embed = nn.Embedding(num_node_types, hidden_dim)

        # Edge type embeddings for attention bias
        self.edge_type_embed = nn.Embedding(num_edge_types, num_heads)

        # Attention projections per head
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Type constraint matrix: which node types can connect via which edge types
        # Shape: [num_node_types, num_node_types, num_edge_types]
        self.register_buffer(
            'type_constraints',
            self._initialize_type_constraints(num_node_types, num_edge_types)
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        logger.debug(f"TypedGraphAttention initialized with {num_heads} heads")

    def _initialize_type_constraints(
        self,
        num_node_types: int,
        num_edge_types: int
    ) -> torch.Tensor:
        """Initialize type constraint matrix.

        Returns tensor of shape [num_node_types, num_node_types, num_edge_types]
        where 1 indicates a valid connection.
        """
        # Default: allow all connections (can be customized)
        constraints = torch.ones(num_node_types, num_node_types, num_edge_types)

        # Example constraints (customize based on domain):
        # - ATTRIBUTE nodes can't cause other nodes
        # - Only STATE nodes have temporal ordering
        # These are soft constraints via attention bias

        return constraints

    def forward(
        self,
        nodes: torch.Tensor,
        node_types: Optional[torch.Tensor] = None,
        edge_types: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply typed graph attention.

        Args:
            nodes: Node features [B, N, hidden_dim]
            node_types: Node type indices [B, N], optional
            edge_types: Edge type indices [B, N, N], optional
            adjacency: Adjacency matrix [B, N, N], optional (1 = connected)

        Returns:
            Tuple of:
                - Updated node features [B, N, hidden_dim]
                - Attention weights [B, num_heads, N, N]
        """
        batch_size, num_nodes, _ = nodes.shape

        # Add node type embeddings if provided
        if node_types is not None:
            type_embeds = self.node_type_embed(node_types)
            nodes = nodes + type_embeds

        # Project to Q, K, V
        Q = self.query_proj(nodes)  # [B, N, H]
        K = self.key_proj(nodes)
        V = self.value_proj(nodes)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, num_heads, N, head_dim]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # [B, num_heads, N, N]

        # Apply edge type bias if provided (5.7)
        if edge_types is not None:
            edge_bias = self.edge_type_embed(edge_types)  # [B, N, N, num_heads]
            edge_bias = edge_bias.permute(0, 3, 1, 2)  # [B, num_heads, N, N]
            scores = scores + edge_bias

        # Apply type constraints as soft mask
        if node_types is not None:
            type_mask = self._compute_type_mask(node_types, edge_types)
            scores = scores + type_mask.unsqueeze(1)  # [B, 1, N, N]

        # Apply adjacency mask if provided
        if adjacency is not None:
            adj_mask = (1 - adjacency) * -1e9
            scores = scores + adj_mask.unsqueeze(1)

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # [B, num_heads, N, head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)

        # Output projection with residual
        output = self.output_proj(context)
        output = self.layer_norm(nodes + output)

        return output, attn_weights

    def _compute_type_mask(
        self,
        node_types: torch.Tensor,
        edge_types: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute soft mask based on type constraints."""
        batch_size, num_nodes = node_types.shape

        # Get type pairs
        src_types = node_types.unsqueeze(2).expand(-1, -1, num_nodes)
        dst_types = node_types.unsqueeze(1).expand(-1, num_nodes, -1)

        # Simple type compatibility mask (can be enhanced)
        # Same type nodes get bonus, different types get small penalty
        type_match = (src_types == dst_types).float()
        mask = type_match * 0.1 - (1 - type_match) * 0.05

        return mask


class RelationalReasoning(nn.Module):
    """Relational reasoning over object representations with Graph Attention (5.7).

    Enhanced with:
    - Typed Graph Attention Networks
    - Type constraints for nodes and edges
    - Multi-layer relational processing

    Attributes:
        object_dim: Input object feature dimension
        relation_dim: Hidden dimension for relational processing
        num_heads: Number of attention heads
        num_layers: Number of GAT layers
        num_node_types: Number of distinct node types
        num_edge_types: Number of distinct edge types
    """

    def __init__(
        self,
        object_dim: int,
        relation_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        num_node_types: int = 5,
        num_edge_types: int = 6,
    ) -> None:
        super().__init__()
        self.object_dim = object_dim
        self.relation_dim = relation_dim
        self.num_layers = num_layers

        # Input projection
        self.object_encoder = nn.Linear(object_dim, relation_dim)

        # Typed Graph Attention layers (5.7)
        self.gat_layers = nn.ModuleList([
            TypedGraphAttention(
                hidden_dim=relation_dim,
                num_heads=num_heads,
                num_node_types=num_node_types,
                num_edge_types=num_edge_types,
            )
            for _ in range(num_layers)
        ])

        # Standard attention layers (fallback for simple cases)
        self.relation_layers = nn.ModuleList([
            nn.MultiheadAttention(relation_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(relation_dim)
            for _ in range(num_layers)
        ])

        self.ffn = nn.Sequential(
            nn.Linear(relation_dim, relation_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(relation_dim * 4, relation_dim)
        )

        self.final_norm = nn.LayerNorm(relation_dim)

        logger.debug(f"RelationalReasoning initialized with {num_layers} GAT layers")

    def forward(
        self,
        objects: torch.Tensor,
        node_types: Optional[torch.Tensor] = None,
        edge_types: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Compute relational representations.

        Args:
            objects: Object features [B, N, object_dim]
            node_types: Optional node type indices [B, N]
            edge_types: Optional edge type indices [B, N, N]
            adjacency: Optional adjacency matrix [B, N, N]
            return_attention: Whether to return attention weights

        Returns:
            Relational representations [B, N, relation_dim]
            If return_attention, also returns list of attention weights
        """
        x = self.object_encoder(objects)
        attention_weights = []

        if node_types is not None or edge_types is not None:
            # Use typed GAT layers (5.7)
            for gat in self.gat_layers:
                x, attn = gat(x, node_types, edge_types, adjacency)
                if return_attention:
                    attention_weights.append(attn)
        else:
            # Fall back to standard attention
            for attn, norm in zip(self.relation_layers, self.layer_norms):
                attn_out, attn_weights_layer = attn(x, x, x)
                x = norm(x + attn_out)
                if return_attention:
                    attention_weights.append(attn_weights_layer)

        # FFN with residual
        x = self.final_norm(x + self.ffn(x))

        if return_attention:
            return x, attention_weights
        return x


class GrangerCausalityTest(nn.Module):
    """Granger causality test for temporal causal inference (5.9).

    Tests whether past values of X help predict Y beyond Y's own past.
    Uses learned regression models instead of traditional VAR.

    Attributes:
        hidden_size: Hidden dimension for regression models
        num_lags: Number of time lags to consider
        significance_threshold: P-value threshold for causality
    """

    def __init__(
        self,
        hidden_size: int,
        num_lags: int = 3,
        significance_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_lags = num_lags
        self.significance_threshold = significance_threshold

        # Restricted model: Y_t ~ Y_{t-1}, ..., Y_{t-k}
        self.restricted_model = nn.Sequential(
            nn.Linear(hidden_size * num_lags, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        # Unrestricted model: Y_t ~ Y_{t-1}, ..., Y_{t-k}, X_{t-1}, ..., X_{t-k}
        self.unrestricted_model = nn.Sequential(
            nn.Linear(hidden_size * num_lags * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        logger.debug(f"GrangerCausalityTest initialized with {num_lags} lags")

    def forward(
        self,
        x_history: torch.Tensor,
        y_history: torch.Tensor,
        y_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Test Granger causality from X to Y.

        Args:
            x_history: History of X [B, num_lags, hidden_size]
            y_history: History of Y [B, num_lags, hidden_size]
            y_target: Target Y to predict [B, hidden_size]

        Returns:
            Tuple of:
                - causality_score: How much X helps predict Y [B]
                - f_statistic: F-statistic for significance [B]
        """
        batch_size = x_history.size(0)

        # Flatten histories
        y_flat = y_history.view(batch_size, -1)
        x_flat = x_history.view(batch_size, -1)

        # Restricted prediction (Y only)
        y_pred_restricted = self.restricted_model(y_flat)

        # Unrestricted prediction (Y and X)
        xy_flat = torch.cat([y_flat, x_flat], dim=-1)
        y_pred_unrestricted = self.unrestricted_model(xy_flat)

        # Compute residual sum of squares
        rss_restricted = ((y_target - y_pred_restricted) ** 2).sum(dim=-1)
        rss_unrestricted = ((y_target - y_pred_unrestricted) ** 2).sum(dim=-1)

        # F-statistic: (RSS_r - RSS_u) / RSS_u * df_u / df_diff
        # Simplified version using ratio of improvements
        epsilon = 1e-8
        improvement = (rss_restricted - rss_unrestricted) / (rss_restricted + epsilon)
        f_statistic = improvement * self.num_lags  # Approximation

        # Causality score: how much X improves prediction
        causality_score = torch.sigmoid(f_statistic * 2)  # Normalized to [0, 1]

        return causality_score, f_statistic


class CausalGraphLearner(nn.Module):
    """Learn causal structure from observations with temporal ordering (5.9).

    Enhanced with:
    - Granger causality testing for temporal data
    - Temporal ordering constraints
    - Time-lagged causal inference

    Attributes:
        num_variables: Number of variables in the system
        hidden_size: Hidden dimension for effect prediction
        threshold: Threshold for edge existence
        num_lags: Number of time lags for Granger causality
        enforce_acyclicity: Whether to enforce DAG constraint
    """

    def __init__(
        self,
        num_variables: int,
        hidden_size: int,
        threshold: float = 0.3,
        num_lags: int = 3,
        enforce_acyclicity: bool = True,
    ) -> None:
        super().__init__()
        self.num_variables = num_variables
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.num_lags = num_lags
        self.enforce_acyclicity = enforce_acyclicity

        # Learnable adjacency (causal structure)
        self.adjacency_logits = nn.Parameter(
            torch.randn(num_variables, num_variables)
        )

        # Temporal ordering scores (5.9)
        # Higher score = earlier in causal order
        self.temporal_order = nn.Parameter(
            torch.randn(num_variables)
        )

        # Effect predictor for instantaneous causation
        self.effect_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Granger causality tester (5.9)
        self.granger_test = GrangerCausalityTest(
            hidden_size=hidden_size,
            num_lags=num_lags,
        )

        # Temporal effect predictor for lagged causation
        self.temporal_effect_predictor = nn.Sequential(
            nn.Linear(hidden_size * (num_lags + 1), hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        logger.debug(f"CausalGraphLearner initialized with {num_variables} variables")

    def get_adjacency_matrix(self, temperature: float = 1.0) -> torch.Tensor:
        """Get differentiable adjacency matrix with temporal ordering (5.9).

        Returns:
            Adjacency matrix with temporal constraints applied
        """
        probs = torch.sigmoid(self.adjacency_logits / temperature)

        # Remove self-loops
        mask = torch.eye(self.num_variables, device=probs.device)
        probs = probs * (1 - mask)

        # Apply temporal ordering constraint (5.9)
        if self.enforce_acyclicity:
            # Only allow edges from earlier to later in temporal order
            order_scores = self.temporal_order.unsqueeze(1) - self.temporal_order.unsqueeze(0)
            # order_scores[i, j] > 0 means i is before j
            temporal_mask = torch.sigmoid(order_scores * 5)  # Soft mask
            probs = probs * temporal_mask

        return probs

    def get_temporal_order(self) -> torch.Tensor:
        """Get sorted indices representing temporal causal order.

        Returns:
            Indices sorted by temporal order (earliest first)
        """
        return torch.argsort(self.temporal_order, descending=True)

    def forward(
        self,
        variables: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute causal effects.

        Args:
            variables: Variable states [B, num_vars, hidden_size]
            temperature: Temperature for adjacency softmax

        Returns:
            Tuple of:
                - effects: Predicted effects [B, num_vars, hidden_size]
                - adjacency: Learned adjacency matrix [num_vars, num_vars]
        """
        adjacency = self.get_adjacency_matrix(temperature)

        batch_size = variables.size(0)
        num_vars = variables.size(1)

        effects = []
        for i in range(num_vars):
            causes = []
            for j in range(num_vars):
                if i != j:
                    combined = torch.cat([variables[:, j], variables[:, i]], dim=-1)
                    effect = self.effect_predictor(combined)
                    weighted_effect = effect * adjacency[j, i].unsqueeze(0).unsqueeze(-1)
                    causes.append(weighted_effect)

            if causes:
                total_effect = torch.stack(causes, dim=1).sum(dim=1)
            else:
                total_effect = torch.zeros_like(variables[:, i])

            effects.append(total_effect)

        effects_tensor = torch.stack(effects, dim=1)

        return effects_tensor, adjacency

    def forward_temporal(
        self,
        variable_history: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute causal effects with temporal information (5.9).

        Args:
            variable_history: Variable states over time [B, T, num_vars, hidden_size]
            temperature: Temperature for adjacency softmax

        Returns:
            Tuple of:
                - effects: Predicted effects at final time [B, num_vars, hidden_size]
                - adjacency: Instantaneous adjacency [num_vars, num_vars]
                - granger_matrix: Granger causality matrix [num_vars, num_vars]
        """
        batch_size, time_steps, num_vars, hidden_size = variable_history.shape

        if time_steps < self.num_lags + 1:
            # Not enough history for Granger causality
            logger.warning(f"Insufficient history for Granger test: {time_steps} < {self.num_lags + 1}")
            current = variable_history[:, -1]
            effects, adjacency = self.forward(current, temperature)
            granger_matrix = torch.zeros(num_vars, num_vars, device=variable_history.device)
            return effects, adjacency, granger_matrix

        # Instantaneous effects
        current_vars = variable_history[:, -1]  # [B, num_vars, hidden_size]
        instantaneous_effects, adjacency = self.forward(current_vars, temperature)

        # Granger causality matrix (5.9)
        granger_matrix = torch.zeros(num_vars, num_vars, device=variable_history.device)

        for i in range(num_vars):
            for j in range(num_vars):
                if i != j:
                    # Test if j Granger-causes i
                    x_history = variable_history[:, -(self.num_lags + 1):-1, j]  # [B, num_lags, H]
                    y_history = variable_history[:, -(self.num_lags + 1):-1, i]  # [B, num_lags, H]
                    y_target = variable_history[:, -1, i]  # [B, H]

                    causality_score, _ = self.granger_test(x_history, y_history, y_target)
                    granger_matrix[j, i] = causality_score.mean()

        # Combine instantaneous and temporal effects
        temporal_effects = []
        for i in range(num_vars):
            # Collect lagged causes
            lagged_inputs = [current_vars[:, i]]
            for j in range(num_vars):
                if i != j and granger_matrix[j, i] > self.threshold:
                    # Include lagged values from Granger-causing variables
                    lagged = variable_history[:, -(self.num_lags + 1):-1, j].view(batch_size, -1)
                    lagged_inputs.append(lagged[:, :self.hidden_size])  # Truncate

            # Predict with temporal context
            if len(lagged_inputs) > 1:
                # Pad to expected size
                combined = torch.cat(lagged_inputs[:self.num_lags + 1], dim=-1)
                expected_size = self.hidden_size * (self.num_lags + 1)
                if combined.size(-1) < expected_size:
                    combined = F.pad(combined, (0, expected_size - combined.size(-1)))
                combined = combined[:, :expected_size]
                temporal_effect = self.temporal_effect_predictor(combined)
            else:
                temporal_effect = current_vars[:, i]

            temporal_effects.append(temporal_effect)

        effects_tensor = torch.stack(temporal_effects, dim=1)

        return effects_tensor, adjacency, granger_matrix

    def intervene(
        self,
        variables: torch.Tensor,
        intervention_var: int,
        intervention_value: torch.Tensor
    ) -> torch.Tensor:
        """Simulate intervention on causal graph.

        Args:
            variables: Current variable states [B, num_vars, hidden_size]
            intervention_var: Index of variable to intervene on
            intervention_value: Value to set [B, hidden_size]

        Returns:
            Variables after intervention propagation
        """
        adjacency = self.get_adjacency_matrix(temperature=0.1)

        intervened = variables.clone()
        intervened[:, intervention_var] = intervention_value

        # Get temporal order for propagation
        order = self.get_temporal_order()

        # Propagate effects in temporal order
        for var_idx in order:
            if var_idx == intervention_var:
                continue

            if adjacency[intervention_var, var_idx] > self.threshold:
                combined = torch.cat([
                    intervention_value,
                    variables[:, var_idx]
                ], dim=-1)
                effect = self.effect_predictor(combined)
                intervened[:, var_idx] = effect

        return intervened


class AnalogyMaker(nn.Module):
    """Make analogies between concepts with dimension checking (5.8).

    Enhanced with:
    - Explicit dimension checking before all operations
    - Automatic projection for mismatched dimensions
    - Robust error handling for edge cases

    Attributes:
        hidden_size: Expected dimension for concepts
        auto_project: Whether to automatically project mismatched dimensions
    """

    def __init__(
        self,
        hidden_size: int,
        auto_project: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.auto_project = auto_project

        # Relation extractor
        self.relation_extractor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Analogy scorer
        self.analogy_scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Dimension projection layers (5.8)
        # Lazy initialization for different input dimensions
        self._projection_cache: Dict[int, nn.Linear] = {}

        logger.debug(f"AnalogyMaker initialized with hidden_size={hidden_size}")

    def _check_and_project(
        self,
        tensor: torch.Tensor,
        name: str = "tensor"
    ) -> torch.Tensor:
        """Check dimensions and project if necessary (5.8).

        Args:
            tensor: Input tensor to check
            name: Name for logging/error messages

        Returns:
            Tensor with correct dimension

        Raises:
            ValueError: If dimension mismatch and auto_project=False
        """
        if tensor.dim() == 0:
            raise ValueError(f"{name} is a scalar, expected vector of dim {self.hidden_size}")

        actual_dim = tensor.size(-1)

        if actual_dim == self.hidden_size:
            return tensor

        if not self.auto_project:
            raise ValueError(
                f"Dimension mismatch for {name}: expected {self.hidden_size}, got {actual_dim}. "
                f"Set auto_project=True to enable automatic projection."
            )

        # Create or retrieve projection layer
        if actual_dim not in self._projection_cache:
            self._projection_cache[actual_dim] = nn.Linear(
                actual_dim, self.hidden_size, device=tensor.device
            )
            logger.debug(f"Created projection layer for {name}: {actual_dim} -> {self.hidden_size}")

        projection = self._projection_cache[actual_dim]

        # Move to correct device if needed
        if projection.weight.device != tensor.device:
            projection = projection.to(tensor.device)
            self._projection_cache[actual_dim] = projection

        return projection(tensor)

    def _ensure_2d(self, tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        """Ensure tensor is 2D [batch, dim] (5.8).

        Args:
            tensor: Input tensor
            name: Name for logging

        Returns:
            2D tensor
        """
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        elif tensor.dim() == 2:
            return tensor
        elif tensor.dim() > 2:
            # Flatten all but last dimension
            return tensor.view(-1, tensor.size(-1))
        else:
            raise ValueError(f"{name} has invalid dimensions: {tensor.shape}")

    def extract_relation(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Extract relation vector from source to target.

        Args:
            source: Source concept [*, hidden_size]
            target: Target concept [*, hidden_size]

        Returns:
            Relation vector [*, hidden_size]
        """
        # Dimension checking (5.8)
        source = self._check_and_project(source, "source")
        target = self._check_and_project(target, "target")

        # Ensure compatible shapes
        if source.shape[:-1] != target.shape[:-1]:
            # Try to broadcast
            source = self._ensure_2d(source, "source")
            target = self._ensure_2d(target, "target")

            if source.size(0) != target.size(0):
                if source.size(0) == 1:
                    source = source.expand(target.size(0), -1)
                elif target.size(0) == 1:
                    target = target.expand(source.size(0), -1)
                else:
                    raise ValueError(
                        f"Cannot broadcast source shape {source.shape} with target shape {target.shape}"
                    )

        combined = torch.cat([source, target], dim=-1)
        return self.relation_extractor(combined)

    def apply_relation(
        self,
        concept: torch.Tensor,
        relation: torch.Tensor
    ) -> torch.Tensor:
        """Apply relation to concept.

        Args:
            concept: Base concept [*, hidden_size]
            relation: Relation to apply [*, hidden_size]

        Returns:
            Transformed concept [*, hidden_size]
        """
        # Dimension checking (5.8)
        concept = self._check_and_project(concept, "concept")
        relation = self._check_and_project(relation, "relation")

        # Handle shape broadcasting
        if concept.shape != relation.shape:
            concept = self._ensure_2d(concept, "concept")
            relation = self._ensure_2d(relation, "relation")

            if concept.size(0) != relation.size(0):
                if concept.size(0) == 1:
                    concept = concept.expand(relation.size(0), -1)
                elif relation.size(0) == 1:
                    relation = relation.expand(concept.size(0), -1)

        return concept + relation

    def score_analogy(
        self,
        source_a: torch.Tensor,
        source_b: torch.Tensor,
        target_a: torch.Tensor,
        target_b: torch.Tensor
    ) -> torch.Tensor:
        """Score A:B::C:D analogy.

        Args:
            source_a: First source concept
            source_b: Second source concept
            target_a: First target concept
            target_b: Second target concept

        Returns:
            Analogy score (higher = better analogy)
        """
        # Dimension checking for all inputs (5.8)
        source_a = self._check_and_project(source_a, "source_a")
        source_b = self._check_and_project(source_b, "source_b")
        target_a = self._check_and_project(target_a, "target_a")
        target_b = self._check_and_project(target_b, "target_b")

        relation_ab = self.extract_relation(source_a, source_b)
        relation_cd = self.extract_relation(target_a, target_b)

        combined = torch.cat([relation_ab, relation_cd], dim=-1)
        return self.analogy_scorer(combined).squeeze(-1)

    def complete_analogy(
        self,
        source_a: torch.Tensor,
        source_b: torch.Tensor,
        target_a: torch.Tensor,
        candidates: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find best D for A:B::C:D with dimension checking (5.8).

        Args:
            source_a: First source concept
            source_b: Second source concept
            target_a: First target concept
            candidates: Candidate completions [num_candidates, hidden_size]

        Returns:
            Tuple of:
                - best_candidate: Best matching concept
                - similarity: Similarity score of best match
        """
        # Dimension checking (5.8)
        source_a = self._check_and_project(source_a, "source_a")
        source_b = self._check_and_project(source_b, "source_b")
        target_a = self._check_and_project(target_a, "target_a")
        candidates = self._check_and_project(candidates, "candidates")

        # Extract and apply relation
        relation_ab = self.extract_relation(source_a, source_b)
        predicted_d = self.apply_relation(target_a, relation_ab)

        # Ensure proper shapes for similarity computation
        predicted_d = self._ensure_2d(predicted_d, "predicted_d")
        candidates = self._ensure_2d(candidates, "candidates")

        # Handle batch dimension for predicted_d
        if predicted_d.size(0) > 1:
            # Multiple predictions - use mean
            predicted_d = predicted_d.mean(dim=0, keepdim=True)

        # Compute similarities
        # predicted_d: [1, hidden_size]
        # candidates: [num_candidates, hidden_size]
        similarities = F.cosine_similarity(
            predicted_d.expand(candidates.size(0), -1),
            candidates,
            dim=-1
        )

        best_idx = torch.argmax(similarities)
        return candidates[best_idx], similarities[best_idx]

    def find_analogies(
        self,
        concepts: torch.Tensor,
        query_pair: Tuple[int, int],
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find best analogies for a query pair from a set of concepts.

        Args:
            concepts: Set of concepts [N, hidden_size]
            query_pair: Indices (a, b) defining the query relation
            top_k: Number of analogies to return

        Returns:
            Tuple of:
                - indices: Indices of best analogy pairs [top_k, 2]
                - scores: Analogy scores [top_k]
        """
        concepts = self._check_and_project(concepts, "concepts")
        concepts = self._ensure_2d(concepts, "concepts")

        n_concepts = concepts.size(0)
        a_idx, b_idx = query_pair

        # Extract query relation
        query_relation = self.extract_relation(concepts[a_idx], concepts[b_idx])

        # Score all pairs as potential analogies
        scores = []
        pairs = []

        for i in range(n_concepts):
            for j in range(n_concepts):
                if i != j and (i, j) != query_pair:
                    score = self.score_analogy(
                        concepts[a_idx],
                        concepts[b_idx],
                        concepts[i],
                        concepts[j]
                    )
                    scores.append(score)
                    pairs.append([i, j])

        if not scores:
            return torch.tensor([]).long(), torch.tensor([])

        scores_tensor = torch.stack(scores)
        pairs_tensor = torch.tensor(pairs, device=concepts.device)

        # Get top-k
        top_k = min(top_k, len(scores))
        top_scores, top_indices = torch.topk(scores_tensor, top_k)

        return pairs_tensor[top_indices], top_scores


class AbstractReasoner(nn.Module):
    """Unified abstract reasoning module with explicit modes (5.6).

    Enhanced with:
    - Explicit reasoning modes (causal, analogical, relational, counterfactual)
    - Mode-specific processing pipelines
    - Learned mode routing when mode="auto"

    Attributes:
        hidden_size: Hidden dimension for all components
        num_variables: Number of variables for causal reasoning
        num_relation_layers: Number of layers in relational reasoning
    """

    def __init__(
        self,
        hidden_size: int,
        num_variables: int = 10,
        num_relation_layers: int = 2,
        num_node_types: int = 5,
        num_edge_types: int = 6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_variables = num_variables

        # Relational reasoning with GAT (5.7)
        self.relational_reasoning = RelationalReasoning(
            object_dim=hidden_size,
            relation_dim=hidden_size,
            num_layers=num_relation_layers,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
        )

        # Causal reasoning with Granger causality (5.9)
        self.causal_learner = CausalGraphLearner(
            num_variables=num_variables,
            hidden_size=hidden_size
        )

        # Analogy reasoning with dimension checking (5.8)
        self.analogy_maker = AnalogyMaker(hidden_size)

        # Mode router for automatic mode selection (5.6)
        self.reasoning_router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4),  # 4 modes: relational, causal, analogical, counterfactual
            nn.Softmax(dim=-1)
        )

        # Mode names for routing output
        self.mode_names = ["relational", "causal", "analogical", "counterfactual"]

        # Analogy context encoders (5.6)
        self.analogy_context_encoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Output aggregator for combining different reasoning outputs
        self.output_aggregator = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        logger.debug(f"AbstractReasoner initialized with explicit modes")

    def forward(
        self,
        query: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mode: str = "auto",
        analogy_pairs: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        node_types: Optional[torch.Tensor] = None,
        edge_types: Optional[torch.Tensor] = None,
        variable_history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Perform abstract reasoning with explicit mode selection (5.6).

        Args:
            query: Query representation [B, hidden_size]
            context: Context objects [B, N, hidden_size], optional
            mode: Reasoning mode - "auto", "relational", "causal", "analogical", "counterfactual"
            analogy_pairs: For analogical mode - (source_a, source_b, target_a) tensors
            node_types: Node types for typed graph attention [B, N]
            edge_types: Edge types for typed graph attention [B, N, N]
            variable_history: For temporal causal reasoning [B, T, num_vars, hidden_size]

        Returns:
            Dictionary with mode-specific outputs and routing information
        """
        outputs = {}
        batch_size = query.size(0)

        # Mode routing (5.6)
        if mode == "auto":
            routing = self.reasoning_router(query)
            outputs["routing"] = routing
            outputs["mode_names"] = self.mode_names

            # Get dominant mode
            dominant_mode_idx = routing.argmax(dim=-1)
            dominant_mode = self.mode_names[dominant_mode_idx[0].item()]
            outputs["dominant_mode"] = dominant_mode
        else:
            # Validate mode
            valid_modes = ["auto", "relational", "causal", "analogical", "counterfactual"]
            if mode not in valid_modes:
                logger.warning(f"Unknown mode '{mode}', falling back to 'auto'")
                mode = "auto"
            outputs["dominant_mode"] = mode

        # Process based on mode (5.6)
        should_run_relational = mode in ("auto", "relational")
        should_run_causal = mode in ("auto", "causal")
        should_run_analogical = mode in ("auto", "analogical")
        should_run_counterfactual = mode in ("auto", "counterfactual")

        # Relational reasoning (5.7)
        if should_run_relational and context is not None:
            relational_output, attn_weights = self.relational_reasoning(
                context,
                node_types=node_types,
                edge_types=edge_types,
                return_attention=True,
            )
            outputs["relational"] = relational_output
            outputs["relational_attention"] = attn_weights

        # Causal reasoning (5.9)
        if should_run_causal and context is not None and context.size(1) >= 2:
            if variable_history is not None:
                # Use temporal causal inference with Granger causality
                causal_effects, adjacency, granger = self.causal_learner.forward_temporal(
                    variable_history
                )
                outputs["causal_effects"] = causal_effects
                outputs["causal_graph"] = adjacency
                outputs["granger_causality"] = granger
                outputs["temporal_order"] = self.causal_learner.get_temporal_order()
            else:
                # Standard causal inference
                causal_effects, adjacency = self.causal_learner(context)
                outputs["causal_effects"] = causal_effects
                outputs["causal_graph"] = adjacency

        # Analogical reasoning (5.6, 5.8)
        if should_run_analogical:
            if analogy_pairs is not None:
                source_a, source_b, target_a = analogy_pairs

                # Extract relation
                relation = self.analogy_maker.extract_relation(source_a, source_b)
                outputs["analogy_relation"] = relation

                # Apply relation to predict target_b
                predicted_b = self.analogy_maker.apply_relation(target_a, relation)
                outputs["analogy_prediction"] = predicted_b

                # If we have context, use as candidates
                if context is not None:
                    context_2d = context.view(-1, context.size(-1))
                    best_match, score = self.analogy_maker.complete_analogy(
                        source_a, source_b, target_a, context_2d
                    )
                    outputs["analogy_best_match"] = best_match
                    outputs["analogy_score"] = score

            elif context is not None and context.size(1) >= 4:
                # Auto-discover analogies in context
                context_2d = context[0]  # Use first batch item
                pairs, scores = self.analogy_maker.find_analogies(
                    context_2d,
                    query_pair=(0, 1),  # Use first two objects as query
                    top_k=3
                )
                outputs["discovered_analogies"] = pairs
                outputs["analogy_scores"] = scores

        # Counterfactual reasoning placeholder (handled by CounterfactualReasoner)
        if should_run_counterfactual and context is not None:
            # Store context for counterfactual reasoning
            outputs["counterfactual_context"] = context
            outputs["counterfactual_ready"] = True

        # Aggregate outputs if multiple modes ran
        if mode == "auto" and len(outputs) > 3:
            aggregated = self._aggregate_outputs(outputs, batch_size)
            outputs["aggregated"] = aggregated

        return outputs

    def _aggregate_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        batch_size: int
    ) -> torch.Tensor:
        """Aggregate outputs from multiple reasoning modes.

        Args:
            outputs: Dictionary of outputs from different modes
            batch_size: Batch size for padding

        Returns:
            Aggregated representation [B, hidden_size]
        """
        components = []

        # Relational output
        if "relational" in outputs:
            rel = outputs["relational"]
            if rel.dim() == 3:
                rel = rel.mean(dim=1)  # Average over objects
            components.append(rel)
        else:
            components.append(torch.zeros(batch_size, self.hidden_size, device=outputs.get("routing", torch.zeros(1)).device))

        # Causal output
        if "causal_effects" in outputs:
            causal = outputs["causal_effects"]
            if causal.dim() == 3:
                causal = causal.mean(dim=1)
            components.append(causal)
        else:
            components.append(torch.zeros(batch_size, self.hidden_size, device=outputs.get("routing", torch.zeros(1)).device))

        # Analogical output
        if "analogy_prediction" in outputs:
            analogy = outputs["analogy_prediction"]
            if analogy.dim() == 1:
                analogy = analogy.unsqueeze(0).expand(batch_size, -1)
            components.append(analogy)
        else:
            components.append(torch.zeros(batch_size, self.hidden_size, device=outputs.get("routing", torch.zeros(1)).device))

        # Counterfactual placeholder
        if "counterfactual_context" in outputs:
            cf = outputs["counterfactual_context"]
            if cf.dim() == 3:
                cf = cf.mean(dim=1)
            components.append(cf)
        else:
            components.append(torch.zeros(batch_size, self.hidden_size, device=outputs.get("routing", torch.zeros(1)).device))

        # Concatenate and aggregate
        combined = torch.cat(components, dim=-1)
        return self.output_aggregator(combined)

    def reason_causal(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        variable_history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Explicit causal reasoning mode (5.6).

        Args:
            query: Query representation
            context: Context variables
            variable_history: Optional temporal history for Granger causality

        Returns:
            Causal reasoning outputs
        """
        return self.forward(
            query, context, mode="causal", variable_history=variable_history
        )

    def reason_analogical(
        self,
        query: torch.Tensor,
        source_a: torch.Tensor,
        source_b: torch.Tensor,
        target_a: torch.Tensor,
        candidates: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Explicit analogical reasoning mode (5.6).

        Args:
            query: Query representation
            source_a: First source concept
            source_b: Second source concept
            target_a: Target concept to complete analogy for
            candidates: Optional candidate completions

        Returns:
            Analogical reasoning outputs
        """
        return self.forward(
            query,
            context=candidates,
            mode="analogical",
            analogy_pairs=(source_a, source_b, target_a),
        )


class CounterfactualReasoner(nn.Module):
    """Counterfactual reasoning for what-if scenarios with plausibility scoring (5.10).

    Enhanced with:
    - Plausibility scoring for counterfactuals
    - Learned prior over interventions
    - Ranking mechanism for multiple counterfactuals

    Attributes:
        world_model: World model for state transitions
        hidden_size: Hidden dimension
        intervention_dim: Dimension of interventions (projected to hidden_size)
        num_counterfactuals: Number of counterfactuals to generate
    """

    def __init__(
        self,
        world_model: nn.Module,
        hidden_size: int,
        intervention_dim: Optional[int] = None,
        num_counterfactuals: int = 5,
    ) -> None:
        super().__init__()
        self.world_model = world_model
        self.hidden_size = hidden_size
        self.intervention_dim = intervention_dim
        self.num_counterfactuals = num_counterfactuals

        # Lazy initialization of intervention projection (set on first forward)
        self._intervention_proj: Optional[nn.Linear] = None

        self.intervention_encoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.outcome_comparator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Plausibility scoring components (5.10)
        # Learned prior over interventions
        self.intervention_prior = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # Plausibility scorer: considers intervention, state, and outcome
        self.plausibility_scorer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # Counterfactual generator for diverse counterfactuals
        self.counterfactual_generator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * num_counterfactuals),
        )

        # Contrastive loss components for training
        self.contrast_encoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        logger.debug(f"CounterfactualReasoner initialized with plausibility scoring")

    def _project_intervention(self, intervention: torch.Tensor) -> torch.Tensor:
        """Project intervention to hidden_size dimension if needed."""
        intervention_dim = intervention.size(-1)

        if intervention_dim == self.hidden_size:
            return intervention

        # Lazy initialization of projection layer
        if self._intervention_proj is None or self._intervention_proj.in_features != intervention_dim:
            self._intervention_proj = nn.Linear(
                intervention_dim, self.hidden_size, device=intervention.device
            )

        return self._intervention_proj(intervention)

    def compute_intervention_prior(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute prior probability of interventions given state (5.10).

        The prior captures which interventions are more likely/reasonable
        in a given context.

        Args:
            state: Current state [B, hidden_size]

        Returns:
            Log prior probability [B, 1]
        """
        return self.intervention_prior(state)

    def score_plausibility(
        self,
        factual_state: torch.Tensor,
        intervention: torch.Tensor,
        counterfactual_state: torch.Tensor,
    ) -> torch.Tensor:
        """Score the plausibility of a counterfactual (5.10).

        Plausibility considers:
        - How likely the intervention is (prior)
        - How consistent the outcome is with the world model
        - How "close" the counterfactual is to the factual world

        Args:
            factual_state: Original state [B, hidden_size]
            intervention: Applied intervention [B, hidden_size]
            counterfactual_state: Resulting counterfactual state [B, hidden_size]

        Returns:
            Plausibility score in [0, 1] where higher is more plausible
        """
        # Project intervention if needed
        intervention_proj = self._project_intervention(intervention)

        # Combine all information for plausibility scoring
        combined = torch.cat([
            factual_state,
            intervention_proj,
            counterfactual_state
        ], dim=-1)

        # Base plausibility from neural scorer
        base_plausibility = self.plausibility_scorer(combined)

        # Prior contribution
        prior_score = torch.sigmoid(self.compute_intervention_prior(factual_state))

        # Combine base plausibility with prior (weighted average)
        plausibility = 0.7 * base_plausibility + 0.3 * prior_score

        return plausibility.squeeze(-1)

    def generate_counterfactual(
        self,
        factual_state: torch.Tensor,
        intervention: torch.Tensor
    ) -> torch.Tensor:
        """Generate counterfactual outcome.

        Args:
            factual_state: [B, hidden_size] current state representation
            intervention: [B, intervention_dim] intervention (e.g., action probs)
                          Will be projected to hidden_size if dimensions differ

        Returns:
            counterfactual_state: [B, hidden_size] predicted state after intervention
        """
        # Project intervention to match hidden_size if needed
        intervention_proj = self._project_intervention(intervention)

        intervention_encoding = self.intervention_encoder(
            torch.cat([factual_state, intervention_proj], dim=-1)
        )

        counterfactual_state = factual_state + intervention_encoding

        return counterfactual_state

    def generate_multiple_counterfactuals(
        self,
        factual_state: torch.Tensor,
        base_intervention: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate multiple diverse counterfactuals (5.10).

        Args:
            factual_state: Current state [B, hidden_size]
            base_intervention: Optional base intervention to vary from

        Returns:
            Tuple of:
                - counterfactuals: Multiple counterfactual states [B, num_cf, hidden_size]
                - interventions: Corresponding interventions [B, num_cf, hidden_size]
        """
        batch_size = factual_state.size(0)

        if base_intervention is None:
            # Generate interventions from state
            base_intervention = torch.zeros(batch_size, self.hidden_size, device=factual_state.device)

        # Project base intervention
        base_intervention_proj = self._project_intervention(base_intervention)

        # Generate diverse interventions
        combined = torch.cat([factual_state, base_intervention_proj], dim=-1)
        intervention_variations = self.counterfactual_generator(combined)

        # Reshape to [B, num_cf, hidden_size]
        interventions = intervention_variations.view(
            batch_size, self.num_counterfactuals, self.hidden_size
        )

        # Generate counterfactual for each intervention
        counterfactuals = []
        for i in range(self.num_counterfactuals):
            cf = self.generate_counterfactual(factual_state, interventions[:, i])
            counterfactuals.append(cf)

        counterfactuals_tensor = torch.stack(counterfactuals, dim=1)

        return counterfactuals_tensor, interventions

    def rank_counterfactuals(
        self,
        factual_state: torch.Tensor,
        counterfactuals: torch.Tensor,
        interventions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rank counterfactuals by plausibility (5.10).

        Args:
            factual_state: Original state [B, hidden_size]
            counterfactuals: Counterfactual states [B, num_cf, hidden_size]
            interventions: Corresponding interventions [B, num_cf, hidden_size]

        Returns:
            Tuple of:
                - ranked_indices: Indices sorted by plausibility [B, num_cf]
                - plausibility_scores: Scores for each counterfactual [B, num_cf]
        """
        batch_size, num_cf, _ = counterfactuals.shape

        scores = []
        for i in range(num_cf):
            score = self.score_plausibility(
                factual_state,
                interventions[:, i],
                counterfactuals[:, i]
            )
            scores.append(score)

        plausibility_scores = torch.stack(scores, dim=1)  # [B, num_cf]

        # Sort by plausibility (descending)
        ranked_indices = torch.argsort(plausibility_scores, dim=-1, descending=True)

        return ranked_indices, plausibility_scores

    def compare_outcomes(
        self,
        factual: torch.Tensor,
        counterfactual: torch.Tensor
    ) -> torch.Tensor:
        """Compare factual vs counterfactual outcomes."""
        combined = torch.cat([factual, counterfactual], dim=-1)
        return self.outcome_comparator(combined).squeeze(-1)

    def forward(
        self,
        factual_state: torch.Tensor,
        intervention: Optional[torch.Tensor] = None,
        return_plausibility: bool = True,
        generate_multiple: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate and evaluate counterfactuals.

        Args:
            factual_state: Current state [B, hidden_size]
            intervention: Optional specific intervention [B, intervention_dim]
            return_plausibility: Whether to compute plausibility scores
            generate_multiple: Whether to generate multiple counterfactuals

        Returns:
            Dictionary with counterfactual outputs
        """
        outputs = {}

        if generate_multiple:
            # Generate multiple diverse counterfactuals
            counterfactuals, interventions = self.generate_multiple_counterfactuals(
                factual_state, intervention
            )
            outputs["counterfactuals"] = counterfactuals
            outputs["interventions"] = interventions

            if return_plausibility:
                ranked_indices, scores = self.rank_counterfactuals(
                    factual_state, counterfactuals, interventions
                )
                outputs["plausibility_scores"] = scores
                outputs["ranked_indices"] = ranked_indices

                # Get best counterfactual
                best_idx = ranked_indices[:, 0]
                batch_idx = torch.arange(factual_state.size(0), device=factual_state.device)
                outputs["best_counterfactual"] = counterfactuals[batch_idx, best_idx]
                outputs["best_plausibility"] = scores[batch_idx, best_idx]
        else:
            # Single counterfactual
            if intervention is None:
                intervention = torch.zeros_like(factual_state)

            counterfactual = self.generate_counterfactual(factual_state, intervention)
            outputs["counterfactual"] = counterfactual

            if return_plausibility:
                plausibility = self.score_plausibility(
                    factual_state, intervention, counterfactual
                )
                outputs["plausibility"] = plausibility

            # Outcome comparison
            comparison = self.compare_outcomes(factual_state, counterfactual)
            outputs["outcome_difference"] = comparison

        return outputs
