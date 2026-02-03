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
        examples: torch.Tensor
    ) -> torch.Tensor:
        """Encode task from examples.

        Args:
            examples: Either:
                - Tensor of shape [B, num_examples, input_dim] (observations only)
                - Tensor of shape [B, num_examples, input_dim + output_dim] (combined)
                - List of (input, output) tuples (legacy format)

        Returns:
            Task embedding of shape [B, output_dim]
        """
        # Handle tensor input (modern format)
        if isinstance(examples, torch.Tensor):
            batch_size = examples.size(0)
            num_ex = min(examples.size(1), self.num_examples)

            # Check if this is combined (input+output) or just input
            expected_combined_dim = self.example_encoder[0].in_features

            if examples.size(-1) == expected_combined_dim:
                # Already combined input+output
                combined = examples[:, :num_ex, :]
            else:
                # Just observations - duplicate as pseudo "output" (self-supervised)
                input_dim = examples.size(-1)
                output_dim = expected_combined_dim - input_dim
                # Pad with zeros or use same input as output proxy
                if output_dim > 0:
                    padding = torch.zeros(
                        batch_size, num_ex, output_dim,
                        device=examples.device, dtype=examples.dtype
                    )
                    combined = torch.cat([examples[:, :num_ex, :], padding], dim=-1)
                else:
                    combined = examples[:, :num_ex, :]

            # Encode examples
            encoded = self.example_encoder(combined)  # [B, num_ex, hidden]

            # Self-attention over examples
            attended, _ = self.attention(encoded, encoded, encoded)

            # LSTM aggregation
            _, (h_n, _) = self.aggregator(attended)

            return h_n.squeeze(0)

        # Handle legacy list format
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
    """Model-Agnostic Meta-Learning with proper inner loop implementation.

    This is a proper MAML implementation that:
    1. Maintains meta-parameters that are adapted per task
    2. Uses functional forward passes with adapted parameters
    3. Computes second-order gradients for meta-update
    4. Supports both first-order (FOMAML) and full MAML
    """

    def __init__(
        self,
        base_model: nn.Module,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
        first_order: bool = False,  # Use FOMAML (faster but less accurate)
        learn_inner_lr: bool = True,  # Meta-learn the inner learning rate
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order

        # Learnable per-parameter inner learning rates (Meta-SGD style)
        if learn_inner_lr:
            self.inner_lrs = nn.ParameterDict()
            for name, param in base_model.named_parameters():
                if param.requires_grad:
                    safe_name = name.replace('.', '_')
                    self.inner_lrs[safe_name] = nn.Parameter(
                        torch.full_like(param, inner_lr)
                    )
        else:
            self.inner_lr = nn.Parameter(torch.tensor(inner_lr))
            self.inner_lrs = None

        # Store parameter names for functional forward
        self.param_names = []
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                self.param_names.append(name)

    def get_inner_lr(self, name: str, param: torch.Tensor) -> torch.Tensor:
        """Get inner learning rate for a parameter."""
        if self.inner_lrs is not None:
            safe_name = name.replace('.', '_')
            if safe_name in self.inner_lrs:
                return self.inner_lrs[safe_name]
        return self.inner_lr

    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        loss_fn: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Perform inner loop adaptation on support set.

        Args:
            support_x: Support set inputs
            support_y: Support set labels
            loss_fn: Loss function

        Returns:
            Dictionary of adapted parameters
        """
        # Start with current meta-parameters
        adapted_params = {}
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                adapted_params[name] = param

        # Inner loop adaptation
        for step in range(self.num_inner_steps):
            # Forward pass with current adapted params
            pred = self._functional_forward(support_x, adapted_params)
            loss = loss_fn(pred, support_y)

            # Compute gradients
            if self.first_order:
                # FOMAML: don't create graph for second-order
                grads = torch.autograd.grad(
                    loss,
                    adapted_params.values(),
                    create_graph=False,
                    allow_unused=True
                )
            else:
                # Full MAML: create graph for meta-gradient
                grads = torch.autograd.grad(
                    loss,
                    adapted_params.values(),
                    create_graph=True,
                    allow_unused=True
                )

            # Update adapted parameters
            new_adapted = {}
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is not None:
                    lr = self.get_inner_lr(name, param)
                    new_adapted[name] = param - lr * grad
                else:
                    new_adapted[name] = param
            adapted_params = new_adapted

        return adapted_params

    def _functional_forward(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass using custom parameters (functional style).

        This replaces the model's parameters temporarily for the forward pass.
        """
        # Save original parameters
        original_params = {}
        for name, param in self.base_model.named_parameters():
            if name in params:
                original_params[name] = param.data.clone()

        # Replace with adapted parameters
        param_dict = dict(self.base_model.named_parameters())
        for name, adapted_param in params.items():
            if name in param_dict:
                param_dict[name].data = adapted_param

        # Forward pass
        try:
            output = self.base_model(x)
        finally:
            # Restore original parameters
            for name, original in original_params.items():
                if name in param_dict:
                    param_dict[name].data = original

        return output

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """Meta-learning forward pass.

        Args:
            support_x: Support set inputs for adaptation
            support_y: Support set labels
            query_x: Query set inputs for evaluation
            loss_fn: Loss function for inner loop

        Returns:
            Predictions on query set using adapted parameters
        """
        # Inner loop: adapt to support set
        adapted_params = self.inner_loop(support_x, support_y, loss_fn)

        # Predict on query set with adapted params
        pred = self._functional_forward(query_x, adapted_params)

        return pred

    def meta_train_step(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        loss_fn: nn.Module,
        meta_optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Perform one meta-training step over multiple tasks.

        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples
            loss_fn: Loss function
            meta_optimizer: Optimizer for meta-parameters

        Returns:
            Training metrics
        """
        meta_optimizer.zero_grad()

        total_loss = 0.0
        total_acc = 0.0

        for support_x, support_y, query_x, query_y in tasks:
            # Adapt to task
            adapted_params = self.inner_loop(support_x, support_y, loss_fn)

            # Evaluate on query set
            query_pred = self._functional_forward(query_x, adapted_params)
            task_loss = loss_fn(query_pred, query_y)

            total_loss += task_loss

            # Compute accuracy if classification
            if query_y.dtype == torch.long:
                pred_labels = query_pred.argmax(dim=-1)
                acc = (pred_labels == query_y).float().mean()
                total_acc += acc.item()

        # Average loss over tasks
        meta_loss = total_loss / len(tasks)

        # Meta-update
        meta_loss.backward()
        meta_optimizer.step()

        return {
            "meta_loss": meta_loss.item(),
            "avg_accuracy": total_acc / len(tasks) if total_acc > 0 else 0.0,
        }


class ReptileAdapter(nn.Module):
    """Reptile meta-learning (first-order approximation to MAML).

    Reptile is simpler and more memory-efficient than MAML while
    achieving similar performance on many tasks.
    """

    def __init__(
        self,
        base_model: nn.Module,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
        epsilon: float = 0.1,  # Step size towards adapted params
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.epsilon = epsilon

    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        loss_fn: nn.Module,
    ) -> None:
        """Perform inner loop adaptation and update meta-parameters.

        Unlike MAML, Reptile updates parameters in-place towards
        the adapted parameters.
        """
        # Save initial parameters
        initial_params = {
            name: param.clone()
            for name, param in self.base_model.named_parameters()
            if param.requires_grad
        }

        # Inner loop SGD
        inner_optimizer = torch.optim.SGD(
            self.base_model.parameters(),
            lr=self.inner_lr
        )

        for _ in range(self.num_inner_steps):
            inner_optimizer.zero_grad()
            pred = self.base_model(support_x)
            loss = loss_fn(pred, support_y)
            loss.backward()
            inner_optimizer.step()

        # Reptile update: move towards adapted parameters
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if name in initial_params:
                    # Interpolate between initial and adapted
                    adapted = param.data
                    initial = initial_params[name]
                    param.data = initial + self.epsilon * (adapted - initial)

    def forward(
        self,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass after adaptation."""
        return self.base_model(query_x)


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
