"""Abstract reasoning and relational thinking modules."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class RelationalReasoning(nn.Module):
    """Relational reasoning over object representations."""

    def __init__(
        self,
        object_dim: int,
        relation_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.object_dim = object_dim
        self.relation_dim = relation_dim
        
        self.object_encoder = nn.Linear(object_dim, relation_dim)
        
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
            nn.ReLU(),
            nn.Linear(relation_dim * 4, relation_dim)
        )

    def forward(self, objects: torch.Tensor) -> torch.Tensor:
        """Compute relational representations."""
        x = self.object_encoder(objects)
        
        for attn, norm in zip(self.relation_layers, self.layer_norms):
            attn_out, _ = attn(x, x, x)
            x = norm(x + attn_out)
        
        x = x + self.ffn(x)
        
        return x


class CausalGraphLearner(nn.Module):
    """Learn causal structure from observations."""

    def __init__(
        self,
        num_variables: int,
        hidden_size: int,
        threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_variables = num_variables
        self.threshold = threshold
        
        self.adjacency_logits = nn.Parameter(
            torch.randn(num_variables, num_variables)
        )
        
        self.effect_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def get_adjacency_matrix(self, temperature: float = 1.0) -> torch.Tensor:
        """Get differentiable adjacency matrix."""
        probs = torch.sigmoid(self.adjacency_logits / temperature)
        
        mask = torch.eye(self.num_variables, device=probs.device)
        probs = probs * (1 - mask)
        
        return probs

    def forward(
        self,
        variables: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute causal effects."""
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

    def intervene(
        self,
        variables: torch.Tensor,
        intervention_var: int,
        intervention_value: torch.Tensor
    ) -> torch.Tensor:
        """Simulate intervention on causal graph."""
        adjacency = self.get_adjacency_matrix(temperature=0.1)
        
        intervened = variables.clone()
        intervened[:, intervention_var] = intervention_value
        
        downstream_mask = adjacency[intervention_var] > self.threshold
        
        for var_idx in range(self.num_variables):
            if downstream_mask[var_idx]:
                combined = torch.cat([
                    intervention_value,
                    variables[:, var_idx]
                ], dim=-1)
                effect = self.effect_predictor(combined)
                intervened[:, var_idx] = effect
        
        return intervened


class AnalogyMaker(nn.Module):
    """Make analogies between concepts."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        
        self.relation_extractor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.analogy_scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def extract_relation(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Extract relation vector from source to target."""
        combined = torch.cat([source, target], dim=-1)
        return self.relation_extractor(combined)

    def apply_relation(
        self,
        concept: torch.Tensor,
        relation: torch.Tensor
    ) -> torch.Tensor:
        """Apply relation to concept."""
        return concept + relation

    def score_analogy(
        self,
        source_a: torch.Tensor,
        source_b: torch.Tensor,
        target_a: torch.Tensor,
        target_b: torch.Tensor
    ) -> torch.Tensor:
        """Score A:B::C:D analogy."""
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
        """Find best D for A:B::C:D."""
        relation_ab = self.extract_relation(source_a, source_b)
        
        predicted_d = self.apply_relation(target_a, relation_ab)
        
        # Handle both cases: predicted_d might be batched
        if predicted_d.dim() == 1:
            predicted_d = predicted_d.unsqueeze(0)
        if candidates.dim() == 1:
            candidates = candidates.unsqueeze(0)
        
        # predicted_d: [batch?, hidden_size]
        # candidates: [num_candidates, hidden_size]
        similarities = F.cosine_similarity(
            predicted_d,
            candidates,
            dim=-1
        )
        
        best_idx = torch.argmax(similarities)
        return candidates[best_idx], similarities[best_idx]


class AbstractReasoner(nn.Module):
    """Unified abstract reasoning module."""

    def __init__(
        self,
        hidden_size: int,
        num_variables: int = 10,
        num_relation_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        
        self.relational_reasoning = RelationalReasoning(
            object_dim=hidden_size,
            relation_dim=hidden_size,
            num_layers=num_relation_layers
        )
        
        self.causal_learner = CausalGraphLearner(
            num_variables=num_variables,
            hidden_size=hidden_size
        )
        
        self.analogy_maker = AnalogyMaker(hidden_size)
        
        self.reasoning_router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
            nn.Softmax(dim=-1)
        )

    def forward(
        self,
        query: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mode: str = "auto"
    ) -> Dict[str, torch.Tensor]:
        """Perform abstract reasoning."""
        outputs = {}
        
        if mode == "auto":
            routing = self.reasoning_router(query)
            outputs["routing"] = routing
        
        if context is not None and (mode == "auto" or mode == "relational"):
            relational_output = self.relational_reasoning(context)
            outputs["relational"] = relational_output
        
        if mode == "auto" or mode == "causal":
            if context is not None and context.size(1) >= 2:
                causal_effects, adjacency = self.causal_learner(context)
                outputs["causal_effects"] = causal_effects
                outputs["causal_graph"] = adjacency
        
        return outputs


class CounterfactualReasoner(nn.Module):
    """Counterfactual reasoning for what-if scenarios."""

    def __init__(
        self,
        world_model: nn.Module,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.world_model = world_model
        self.hidden_size = hidden_size
        
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

    def generate_counterfactual(
        self,
        factual_state: torch.Tensor,
        intervention: torch.Tensor
    ) -> torch.Tensor:
        """Generate counterfactual outcome."""
        intervention_encoding = self.intervention_encoder(
            torch.cat([factual_state, intervention], dim=-1)
        )
        
        counterfactual_state = factual_state + intervention_encoding
        
        return counterfactual_state

    def compare_outcomes(
        self,
        factual: torch.Tensor,
        counterfactual: torch.Tensor
    ) -> torch.Tensor:
        """Compare factual vs counterfactual outcomes."""
        combined = torch.cat([factual, counterfactual], dim=-1)
        return self.outcome_comparator(combined).squeeze(-1)
