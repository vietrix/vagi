"""Meta-learning and transfer learning components."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class TaskEmbedding(nn.Module):
    """Embed tasks from few-shot examples."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 256,
        num_examples: int = 5,
    ) -> None:
        super().__init__()
        self.num_examples = num_examples
        
        self.example_encoder = nn.Sequential(
            nn.Linear(input_dim + output_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.aggregator = nn.LSTM(hidden_size, output_dim, batch_first=True)
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)

    def forward(
        self,
        examples: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Encode task from input-output examples."""
        encoded_examples = []
        
        for inp, out in examples[:self.num_examples]:
            combined = torch.cat([inp, out], dim=-1)
            encoded = self.example_encoder(combined)
            encoded_examples.append(encoded)
        
        if not encoded_examples:
            raise ValueError("No examples provided")
        
        stacked = torch.stack(encoded_examples, dim=0).unsqueeze(0)
        
        attended, _ = self.attention(stacked, stacked, stacked)
        
        _, (h_n, _) = self.aggregator(attended)
        
        return h_n.squeeze(0)


class MAMLAdapter(nn.Module):
    """Model-Agnostic Meta-Learning adaptation."""

    def __init__(
        self,
        base_model: nn.Module,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.inner_lr = nn.Parameter(torch.tensor(inner_lr))
        self.num_inner_steps = num_inner_steps
        
        self.meta_params = nn.ParameterDict()
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                self.meta_params[name.replace('.', '_')] = nn.Parameter(
                    param.clone()
                )

    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        loss_fn: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Perform inner loop adaptation."""
        adapted_params = {}
        for name, param in self.meta_params.items():
            adapted_params[name] = param.clone()
        
        for _ in range(self.num_inner_steps):
            with torch.enable_grad():
                pred = self._forward_with_params(support_x, adapted_params)
                loss = loss_fn(pred, support_y)
            
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=True
            )
            
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is not None:
                    adapted_params[name] = param - self.inner_lr * grad
        
        return adapted_params

    def _forward_with_params(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass with custom parameters."""
        return self.base_model(x)

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """Meta-learning forward pass."""
        adapted_params = self.inner_loop(support_x, support_y, loss_fn)
        
        pred = self._forward_with_params(query_x, adapted_params)
        
        return pred


class CurriculumScheduler(nn.Module):
    """Automatic curriculum learning scheduler."""

    def __init__(
        self,
        num_tasks: int,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        
        self.task_difficulty = nn.Parameter(
            torch.linspace(0.1, 1.0, num_tasks)
        )
        
        self.performance_history = []
        
        self.scheduler_net = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_tasks),
            nn.Softmax(dim=-1)
        )

    def update_performance(
        self,
        task_id: int,
        performance: float
    ) -> None:
        """Update task performance history."""
        self.performance_history.append((task_id, performance))
        
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)

    def get_next_task(
        self,
        student_state: torch.Tensor,
        mode: str = "zone_of_proximal"
    ) -> int:
        """Select next task for training."""
        if mode == "random":
            return torch.randint(0, self.num_tasks, (1,)).item()
        
        if mode == "sequential":
            return len(self.performance_history) % self.num_tasks
        
        if mode == "zone_of_proximal":
            recent_perf = self._compute_recent_performance()
            
            difficulties = self.task_difficulty.detach()
            
            too_easy = difficulties < (recent_perf - 0.2)
            too_hard = difficulties > (recent_perf + 0.3)
            
            mask = ~(too_easy | too_hard)
            
            if mask.sum() == 0:
                return torch.randint(0, self.num_tasks, (1,)).item()
            
            valid_tasks = torch.where(mask)[0]
            return valid_tasks[torch.randint(0, len(valid_tasks), (1,))].item()
        
        raise ValueError(f"Unknown mode: {mode}")

    def _compute_recent_performance(self, window: int = 10) -> float:
        """Compute average recent performance."""
        if not self.performance_history:
            return 0.5
        
        recent = self.performance_history[-window:]
        performances = [p for _, p in recent]
        return sum(performances) / len(performances)


class TransferLearner(nn.Module):
    """Transfer learning across domains."""

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        shared_dim: int = 256,
    ) -> None:
        super().__init__()
        
        self.source_encoder = nn.Sequential(
            nn.Linear(source_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim)
        )
        
        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim)
        )
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.ReLU(),
            nn.Linear(shared_dim // 2, 2)
        )
        
        self.shared_predictor = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        domain: str = "source"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input and predict domain."""
        if domain == "source":
            encoded = self.source_encoder(x)
        elif domain == "target":
            encoded = self.target_encoder(x)
        else:
            raise ValueError(f"Unknown domain: {domain}")
        
        domain_pred = self.domain_classifier(encoded)
        
        output = self.shared_predictor(encoded)
        
        return output, domain_pred

    def domain_adaptation_loss(
        self,
        source_x: torch.Tensor,
        target_x: torch.Tensor
    ) -> torch.Tensor:
        """Compute domain adaptation loss."""
        source_encoded = self.source_encoder(source_x)
        target_encoded = self.target_encoder(target_x)
        
        source_domain = self.domain_classifier(source_encoded)
        target_domain = self.domain_classifier(target_encoded)
        
        source_labels = torch.zeros(source_x.size(0), dtype=torch.long, device=source_x.device)
        target_labels = torch.ones(target_x.size(0), dtype=torch.long, device=target_x.device)
        
        loss = F.cross_entropy(source_domain, source_labels) + \
               F.cross_entropy(target_domain, target_labels)
        
        mmd_loss = self._mmd_loss(source_encoded, target_encoded)
        
        return loss + mmd_loss

    def _mmd_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Maximum Mean Discrepancy loss."""
        source_mean = source.mean(dim=0)
        target_mean = target.mean(dim=0)
        
        return F.mse_loss(source_mean, target_mean)


class FewShotLearner(nn.Module):
    """Few-shot learning with prototypical networks."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def compute_prototypes(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor
    ) -> torch.Tensor:
        """Compute class prototypes from support set."""
        embeddings = self.encoder(support_x)
        
        prototypes = []
        for c in range(self.num_classes):
            mask = support_y == c
            if mask.sum() > 0:
                class_embeddings = embeddings[mask]
                prototype = class_embeddings.mean(dim=0)
            else:
                prototype = torch.zeros(embeddings.size(-1), device=embeddings.device)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)

    def forward(
        self,
        query_x: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """Classify queries based on prototypes."""
        query_embeddings = self.encoder(query_x)
        
        distances = torch.cdist(query_embeddings, prototypes)
        
        logits = -distances
        
        return logits
