"""Object-centric scene understanding with scene graphs."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SceneGraph:
    """Structured scene representation."""
    objects: torch.Tensor  # [N, object_dim]
    relations: torch.Tensor  # [N, N, relation_dim]
    object_types: Optional[torch.Tensor] = None  # [N]
    relation_types: Optional[torch.Tensor] = None  # [N, N]
    bounding_boxes: Optional[torch.Tensor] = None  # [N, 4]
    
    def __len__(self) -> int:
        return self.objects.size(0)


class DynamicClassifier(nn.Module):
    """Classifier with support for dynamic class addition at runtime."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 20,
        class_embed_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim

        # Feature projection
        self.feature_proj = nn.Linear(input_dim, class_embed_dim)

        # Class embeddings (learnable prototypes for each class)
        self.class_embeddings = nn.Parameter(
            torch.randn(num_classes, class_embed_dim)
        )
        nn.init.xavier_uniform_(self.class_embeddings)

        # Temperature for softmax
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Track class names for interpretability
        self.class_names: List[str] = [f"class_{i}" for i in range(num_classes)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify inputs using cosine similarity with class embeddings.

        Args:
            x: Input features [B, N, input_dim]

        Returns:
            Logits [B, N, num_classes]
        """
        # Project features
        features = self.feature_proj(x)  # [B, N, class_embed_dim]

        # Normalize for cosine similarity
        features_norm = F.normalize(features, dim=-1)
        class_norm = F.normalize(self.class_embeddings, dim=-1)

        # Compute similarity as logits
        logits = torch.einsum('bnd,cd->bnc', features_norm, class_norm)
        logits = logits / self.temperature

        return logits

    def add_class(
        self,
        class_name: str,
        prototype: Optional[torch.Tensor] = None
    ) -> int:
        """Dynamically add a new class at runtime.

        Args:
            class_name: Name for the new class
            prototype: Optional prototype embedding for the new class

        Returns:
            Index of the new class
        """
        new_class_idx = self.num_classes

        # Create new embedding
        if prototype is None:
            new_embedding = torch.randn(1, self.class_embed_dim, device=self.class_embeddings.device)
            nn.init.xavier_uniform_(new_embedding)
        else:
            new_embedding = prototype.view(1, self.class_embed_dim)

        # Expand class embeddings
        self.class_embeddings = nn.Parameter(
            torch.cat([self.class_embeddings.data, new_embedding], dim=0)
        )

        self.num_classes += 1
        self.class_names.append(class_name)

        return new_class_idx

    def add_classes_from_examples(
        self,
        class_names: List[str],
        examples: torch.Tensor
    ) -> List[int]:
        """Add multiple new classes from example embeddings.

        Args:
            class_names: Names for the new classes
            examples: Example features [num_classes, num_examples, input_dim]

        Returns:
            Indices of the new classes
        """
        new_indices = []

        for i, name in enumerate(class_names):
            # Compute prototype as mean of projected examples
            example_features = self.feature_proj(examples[i])  # [num_examples, class_embed_dim]
            prototype = example_features.mean(dim=0, keepdim=True)
            idx = self.add_class(name, prototype)
            new_indices.append(idx)

        return new_indices


class ObjectDetector(nn.Module):
    """Detect and extract objects from observations with configurable object types."""

    def __init__(
        self,
        obs_dim: int,
        object_dim: int = 128,
        max_objects: int = 10,
        hidden_size: int = 256,
        num_object_types: int = 20,
        class_embed_dim: int = 64,
        adaptive_stopping: bool = True,
    ):
        super().__init__()
        self.max_objects = max_objects
        self.object_dim = object_dim
        self.num_object_types = num_object_types

        # Slot attention for object extraction
        self.slot_attention = SlotAttention(
            num_slots=max_objects,
            slot_dim=object_dim,
            input_dim=obs_dim,
            num_iterations=3,
            adaptive_stopping=adaptive_stopping
        )

        # Object property decoder
        self.property_decoder = nn.Sequential(
            nn.Linear(object_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, object_dim)
        )

        # Dynamic object type classifier (configurable and extensible)
        self.type_classifier = DynamicClassifier(
            input_dim=object_dim,
            num_classes=num_object_types,
            class_embed_dim=class_embed_dim
        )

        # Bounding box predictor for spatial localization
        self.bbox_predictor = nn.Sequential(
            nn.Linear(object_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4),  # [x, y, w, h]
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def forward(
        self,
        obs: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect objects in observation.

        Args:
            obs: [B, obs_dim] or [B, H, W, C]
            return_attention: Whether to return attention weights

        Returns:
            objects: [B, N, object_dim]
            object_types: [B, N, num_types]
            bounding_boxes: [B, N, 4]
            attention: (optional) [B, N, input_positions]
        """
        batch_size = obs.size(0)

        # Flatten spatial dims if needed
        if obs.dim() == 4:
            obs = obs.flatten(1, 2)  # [B, H*W, C]
        elif obs.dim() == 2:
            obs = obs.unsqueeze(1)  # [B, 1, obs_dim]

        # Extract object slots
        slots, attention = self.slot_attention(obs)

        # Decode object properties
        objects = self.property_decoder(slots)

        # Classify object types (using dynamic classifier)
        object_types = self.type_classifier(slots)

        # Predict bounding boxes
        bounding_boxes = self.bbox_predictor(slots)

        if return_attention:
            return objects, object_types, bounding_boxes, attention

        return objects, object_types, bounding_boxes

    def add_object_type(
        self,
        class_name: str,
        prototype: Optional[torch.Tensor] = None
    ) -> int:
        """Add a new object type dynamically.

        Args:
            class_name: Name for the new object type
            prototype: Optional prototype embedding

        Returns:
            Index of the new class
        """
        return self.type_classifier.add_class(class_name, prototype)

    def get_object_type_names(self) -> List[str]:
        """Get list of all object type names."""
        return self.type_classifier.class_names


class SlotAttention(nn.Module):
    """Slot Attention for object-centric learning with adaptive stopping."""

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        input_dim: int,
        num_iterations: int = 3,
        epsilon: float = 1e-8,
        adaptive_stopping: bool = True,
        convergence_threshold: float = 1e-3,
        min_iterations: int = 1,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.adaptive_stopping = adaptive_stopping
        self.convergence_threshold = convergence_threshold
        self.min_iterations = min_iterations

        # Learnable slot initialization
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, slot_dim))

        # Layer norm
        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        # Linear projections for attention
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(input_dim, slot_dim, bias=False)

        # Slot update MLP
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(),
            nn.Linear(slot_dim * 2, slot_dim)
        )

        # Convergence statistics (for monitoring)
        self.register_buffer('avg_iterations_used', torch.tensor(0.0))
        self.register_buffer('convergence_count', torch.tensor(0))

    def _check_convergence(
        self,
        slots: torch.Tensor,
        slots_prev: torch.Tensor
    ) -> bool:
        """Check if slot updates have converged.

        Args:
            slots: Current slot values [B, num_slots, slot_dim]
            slots_prev: Previous slot values [B, num_slots, slot_dim]

        Returns:
            True if converged (change below threshold)
        """
        # Compute relative change in slots
        delta = (slots - slots_prev).abs()
        relative_change = delta / (slots_prev.abs() + self.epsilon)
        max_change = relative_change.max().item()
        return max_change < self.convergence_threshold

    def forward(
        self,
        inputs: torch.Tensor,
        return_iterations: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply slot attention with optional adaptive stopping.

        Args:
            inputs: [B, N, input_dim]
            return_iterations: If True, return number of iterations used

        Returns:
            slots: [B, num_slots, slot_dim]
            attn: [B, num_slots, N]
            iterations_used: (optional) Number of iterations actually used
        """
        batch_size, num_inputs, input_dim = inputs.size()

        # Initialize slots
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = torch.exp(self.slots_log_sigma).expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        # Normalize inputs
        inputs = self.norm_inputs(inputs)

        # Compute keys and values (only once, outside the loop)
        k = self.project_k(inputs)  # [B, N, slot_dim]
        v = self.project_v(inputs)  # [B, N, slot_dim]

        # Iterative attention with adaptive stopping
        iterations_used = 0
        attn = None

        for i in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Compute attention
            q = self.project_q(slots)  # [B, num_slots, slot_dim]

            # Scaled dot-product attention
            scale = self.slot_dim ** -0.5
            attn_logits = torch.einsum('bsd,bnd->bsn', q, k) * scale
            attn = F.softmax(attn_logits, dim=1)  # [B, num_slots, N]

            # Weighted mean
            attn_sum = attn.sum(dim=-1, keepdim=True) + self.epsilon
            updates = torch.einsum('bsn,bnd->bsd', attn / attn_sum, v)

            # Update slots with GRU-style update
            slots = slots_prev + updates
            slots = slots + self.mlp(self.norm_mlp(slots))

            iterations_used = i + 1

            # Check for adaptive stopping (after minimum iterations)
            if (self.adaptive_stopping and
                i >= self.min_iterations - 1 and
                not self.training):  # Only use adaptive stopping during inference
                if self._check_convergence(slots, slots_prev):
                    break

        # Update convergence statistics
        if not self.training:
            self.avg_iterations_used = (
                self.avg_iterations_used * self.convergence_count +
                iterations_used
            ) / (self.convergence_count + 1)
            self.convergence_count = self.convergence_count + 1

        if return_iterations:
            return slots, attn, iterations_used

        return slots, attn


class RelationNetwork(nn.Module):
    """Detect relations between objects."""

    def __init__(
        self,
        object_dim: int,
        relation_dim: int = 64,
        num_relation_types: int = 10,
    ):
        super().__init__()
        self.relation_dim = relation_dim

        # Relation encoder
        self.relation_encoder = nn.Sequential(
            nn.Linear(object_dim * 2, relation_dim * 2),
            nn.ReLU(),
            nn.Linear(relation_dim * 2, relation_dim)
        )

        # Relation type classifier
        self.relation_classifier = nn.Linear(relation_dim, num_relation_types)

        # Relation existence predictor
        self.existence_predictor = nn.Sequential(
            nn.Linear(relation_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        objects: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute pairwise relations.

        Args:
            objects: [B, N, object_dim]

        Returns:
            relations: [B, N, N, relation_dim]
            relation_types: [B, N, N, num_types]
            existence: [B, N, N]
        """
        batch_size, num_objects, object_dim = objects.size()

        # Create all pairs
        obj_i = objects.unsqueeze(2).expand(-1, -1, num_objects, -1)
        obj_j = objects.unsqueeze(1).expand(-1, num_objects, -1, -1)

        # Concatenate pairs
        pairs = torch.cat([obj_i, obj_j], dim=-1)
        pairs = pairs.view(batch_size, num_objects * num_objects, -1)

        # Encode relations
        relations = self.relation_encoder(pairs)
        relations = relations.view(batch_size, num_objects, num_objects, -1)

        # Classify relation types
        relation_types = self.relation_classifier(relations)

        # Predict existence
        existence = self.existence_predictor(relations).squeeze(-1)

        return relations, relation_types, existence


class SpatialRelationNetwork(nn.Module):
    """Network for computing explicit spatial relations from bounding boxes.

    Computes spatial relations like: left_of, right_of, above, below,
    contains, inside, overlaps, near, far, etc.
    """

    # Spatial relation types
    SPATIAL_RELATIONS = [
        'left_of',      # 0: object_i is left of object_j
        'right_of',     # 1: object_i is right of object_j
        'above',        # 2: object_i is above object_j
        'below',        # 3: object_i is below object_j
        'contains',     # 4: object_i contains object_j
        'inside',       # 5: object_i is inside object_j
        'overlaps',     # 6: object_i overlaps with object_j
        'near',         # 7: object_i is near object_j
        'far',          # 8: object_i is far from object_j
        'same_size',    # 9: objects are roughly the same size
        'larger',       # 10: object_i is larger than object_j
        'smaller',      # 11: object_i is smaller than object_j
    ]

    def __init__(
        self,
        hidden_dim: int = 64,
        num_spatial_relations: int = 12,
        distance_threshold: float = 0.1,
        size_tolerance: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_spatial_relations = num_spatial_relations
        self.distance_threshold = distance_threshold
        self.size_tolerance = size_tolerance

        # Learnable spatial feature extractor
        # Input: 8 features (bbox_i x,y,w,h + bbox_j x,y,w,h)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Relation classifier head
        self.relation_head = nn.Linear(hidden_dim, num_spatial_relations)

        # Soft relation predictor (for differentiable training)
        self.soft_relation_head = nn.Sequential(
            nn.Linear(hidden_dim, num_spatial_relations),
            nn.Sigmoid()
        )

    def compute_geometric_features(
        self,
        bboxes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute geometric features from bounding boxes.

        Args:
            bboxes: [B, N, 4] bounding boxes in (x, y, w, h) format

        Returns:
            Dictionary of geometric features
        """
        batch_size, num_objects, _ = bboxes.size()

        # Extract components
        x = bboxes[..., 0]  # [B, N]
        y = bboxes[..., 1]
        w = bboxes[..., 2]
        h = bboxes[..., 3]

        # Compute centers
        cx = x + w / 2
        cy = y + h / 2

        # Compute areas
        areas = w * h

        return {
            'x': x, 'y': y, 'w': w, 'h': h,
            'cx': cx, 'cy': cy, 'areas': areas
        }

    def compute_hard_spatial_relations(
        self,
        bboxes: torch.Tensor
    ) -> torch.Tensor:
        """Compute hard (binary) spatial relations from bounding boxes.

        Args:
            bboxes: [B, N, 4] bounding boxes

        Returns:
            relations: [B, N, N, num_relations] binary relation indicators
        """
        batch_size, num_objects, _ = bboxes.size()
        device = bboxes.device

        geom = self.compute_geometric_features(bboxes)

        # Initialize relations tensor
        relations = torch.zeros(
            batch_size, num_objects, num_objects,
            self.num_spatial_relations, device=device
        )

        # Expand for pairwise computation
        cx_i = geom['cx'].unsqueeze(2)  # [B, N, 1]
        cx_j = geom['cx'].unsqueeze(1)  # [B, 1, N]
        cy_i = geom['cy'].unsqueeze(2)
        cy_j = geom['cy'].unsqueeze(1)

        x_i = geom['x'].unsqueeze(2)
        x_j = geom['x'].unsqueeze(1)
        y_i = geom['y'].unsqueeze(2)
        y_j = geom['y'].unsqueeze(1)
        w_i = geom['w'].unsqueeze(2)
        w_j = geom['w'].unsqueeze(1)
        h_i = geom['h'].unsqueeze(2)
        h_j = geom['h'].unsqueeze(1)

        areas_i = geom['areas'].unsqueeze(2)
        areas_j = geom['areas'].unsqueeze(1)

        # Compute distance between centers
        dist = torch.sqrt((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2 + 1e-8)

        # Compute edges
        right_i = x_i + w_i
        right_j = x_j + w_j
        bottom_i = y_i + h_i
        bottom_j = y_j + h_j

        # left_of: center_x_i < center_x_j and no horizontal overlap
        relations[..., 0] = (cx_i < cx_j) & (right_i <= x_j + self.distance_threshold)

        # right_of: center_x_i > center_x_j and no horizontal overlap
        relations[..., 1] = (cx_i > cx_j) & (x_i >= right_j - self.distance_threshold)

        # above: center_y_i < center_y_j (assuming y increases downward)
        relations[..., 2] = (cy_i < cy_j) & (bottom_i <= y_j + self.distance_threshold)

        # below: center_y_i > center_y_j
        relations[..., 3] = (cy_i > cy_j) & (y_i >= bottom_j - self.distance_threshold)

        # contains: bbox_i fully contains bbox_j
        relations[..., 4] = (
            (x_i <= x_j) & (y_i <= y_j) &
            (right_i >= right_j) & (bottom_i >= bottom_j) &
            (areas_i > areas_j)
        )

        # inside: bbox_i is fully inside bbox_j
        relations[..., 5] = (
            (x_i >= x_j) & (y_i >= y_j) &
            (right_i <= right_j) & (bottom_i <= bottom_j) &
            (areas_i < areas_j)
        )

        # overlaps: bboxes overlap but neither contains the other
        overlap_x = (x_i < right_j) & (right_i > x_j)
        overlap_y = (y_i < bottom_j) & (bottom_i > y_j)
        overlaps = overlap_x & overlap_y
        relations[..., 6] = overlaps & ~relations[..., 4] & ~relations[..., 5]

        # near: distance is small
        avg_size = (torch.sqrt(areas_i) + torch.sqrt(areas_j)) / 2
        relations[..., 7] = dist < avg_size * 0.5

        # far: distance is large
        relations[..., 8] = dist > avg_size * 2.0

        # same_size: areas are similar
        size_ratio = areas_i / (areas_j + 1e-8)
        relations[..., 9] = (size_ratio > 1 - self.size_tolerance) & (size_ratio < 1 + self.size_tolerance)

        # larger: area_i > area_j
        relations[..., 10] = areas_i > areas_j * (1 + self.size_tolerance)

        # smaller: area_i < area_j
        relations[..., 11] = areas_i < areas_j * (1 - self.size_tolerance)

        return relations.float()

    def forward(
        self,
        bboxes: torch.Tensor,
        return_hard_relations: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial relations between objects.

        Args:
            bboxes: [B, N, 4] bounding boxes in (x, y, w, h) format
            return_hard_relations: Also return hard/binary relations

        Returns:
            soft_relations: [B, N, N, num_relations] soft relation scores
            relation_features: [B, N, N, hidden_dim] relation feature vectors
            hard_relations: (optional) [B, N, N, num_relations] binary relations
        """
        batch_size, num_objects, _ = bboxes.size()

        # Create pairwise bbox features
        bbox_i = bboxes.unsqueeze(2).expand(-1, -1, num_objects, -1)  # [B, N, N, 4]
        bbox_j = bboxes.unsqueeze(1).expand(-1, num_objects, -1, -1)  # [B, N, N, 4]
        pair_features = torch.cat([bbox_i, bbox_j], dim=-1)  # [B, N, N, 8]

        # Encode spatial features
        pair_features_flat = pair_features.view(batch_size * num_objects * num_objects, 8)
        spatial_features = self.spatial_encoder(pair_features_flat)
        relation_features = spatial_features.view(batch_size, num_objects, num_objects, self.hidden_dim)

        # Predict soft relations (differentiable)
        soft_relations = self.soft_relation_head(relation_features)

        if return_hard_relations:
            hard_relations = self.compute_hard_spatial_relations(bboxes)
            return soft_relations, relation_features, hard_relations

        return soft_relations, relation_features

    def get_relation_names(self) -> List[str]:
        """Get list of spatial relation names."""
        return self.SPATIAL_RELATIONS.copy()


class TemporalSlotAttention(nn.Module):
    """Temporal Slot Attention for object permanence and tracking across time.

    Extends slot attention to maintain object identity across frames,
    enabling object permanence understanding - knowing objects continue
    to exist even when temporarily occluded.
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        input_dim: int,
        num_iterations: int = 3,
        epsilon: float = 1e-8,
        memory_decay: float = 0.9,
        match_threshold: float = 0.5,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.memory_decay = memory_decay
        self.match_threshold = match_threshold

        # Base slot attention components
        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        # Learnable slot initialization
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, slot_dim))

        # Attention projections
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(input_dim, slot_dim, bias=False)

        # Slot update MLP
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(),
            nn.Linear(slot_dim * 2, slot_dim)
        )

        # Temporal components for object tracking
        # GRU for temporal slot updates
        self.temporal_gru = nn.GRUCell(slot_dim, slot_dim)

        # Slot matching network (for associating slots across frames)
        self.match_key = nn.Linear(slot_dim, slot_dim)
        self.match_query = nn.Linear(slot_dim, slot_dim)

        # Object existence predictor (is the object visible?)
        self.existence_predictor = nn.Sequential(
            nn.Linear(slot_dim, slot_dim // 2),
            nn.ReLU(),
            nn.Linear(slot_dim // 2, 1),
            nn.Sigmoid()
        )

        # Object occlusion predictor (is the object occluded?)
        self.occlusion_predictor = nn.Sequential(
            nn.Linear(slot_dim, slot_dim // 2),
            nn.ReLU(),
            nn.Linear(slot_dim // 2, 1),
            nn.Sigmoid()
        )

        # Velocity predictor for motion prediction during occlusion
        self.velocity_predictor = nn.Sequential(
            nn.Linear(slot_dim * 2, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim)
        )

        # Memory bank for tracking objects (not learnable, state)
        self.register_buffer('slot_memory', None)
        self.register_buffer('slot_velocities', None)
        self.register_buffer('object_ages', None)

    def reset_memory(self, batch_size: int, device: torch.device) -> None:
        """Reset temporal memory for new sequences."""
        self.slot_memory = torch.zeros(batch_size, self.num_slots, self.slot_dim, device=device)
        self.slot_velocities = torch.zeros(batch_size, self.num_slots, self.slot_dim, device=device)
        self.object_ages = torch.zeros(batch_size, self.num_slots, device=device)

    def compute_slot_matching(
        self,
        current_slots: torch.Tensor,
        previous_slots: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute matching between current and previous slots using Hungarian-like attention.

        Args:
            current_slots: [B, N, slot_dim] current frame slots
            previous_slots: [B, N, slot_dim] previous frame slots

        Returns:
            match_scores: [B, N, N] matching scores
            assignment: [B, N] assignment indices (current -> previous)
        """
        # Compute match keys and queries
        keys = self.match_key(previous_slots)  # [B, N, slot_dim]
        queries = self.match_query(current_slots)  # [B, N, slot_dim]

        # Compute similarity matrix
        scale = self.slot_dim ** -0.5
        match_scores = torch.einsum('bnd,bmd->bnm', queries, keys) * scale
        match_probs = F.softmax(match_scores, dim=-1)  # [B, N, N]

        # Get best assignment (greedy for now, could use Hungarian algorithm)
        assignment = match_probs.argmax(dim=-1)  # [B, N]

        return match_probs, assignment

    def predict_occluded_slots(
        self,
        previous_slots: torch.Tensor,
        velocities: torch.Tensor
    ) -> torch.Tensor:
        """Predict slot positions for potentially occluded objects.

        Args:
            previous_slots: [B, N, slot_dim] previous slot states
            velocities: [B, N, slot_dim] estimated velocities

        Returns:
            predicted_slots: [B, N, slot_dim] predicted slot positions
        """
        # Simple linear prediction: new_position = old_position + velocity
        predicted_slots = previous_slots + velocities
        return predicted_slots

    def update_velocities(
        self,
        current_slots: torch.Tensor,
        previous_slots: torch.Tensor,
        match_assignment: torch.Tensor
    ) -> torch.Tensor:
        """Update velocity estimates based on slot movement.

        Args:
            current_slots: [B, N, slot_dim]
            previous_slots: [B, N, slot_dim]
            match_assignment: [B, N] assignment indices

        Returns:
            velocities: [B, N, slot_dim]
        """
        batch_size, num_slots, slot_dim = current_slots.size()

        # Gather matched previous slots
        match_expanded = match_assignment.unsqueeze(-1).expand(-1, -1, slot_dim)
        matched_previous = torch.gather(previous_slots, 1, match_expanded)

        # Compute velocity as difference
        velocity_input = torch.cat([current_slots, matched_previous], dim=-1)
        velocity_flat = velocity_input.view(batch_size * num_slots, -1)
        velocities = self.velocity_predictor(velocity_flat)
        velocities = velocities.view(batch_size, num_slots, slot_dim)

        return velocities

    def forward(
        self,
        inputs: torch.Tensor,
        use_temporal: bool = True,
        return_tracking_info: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Apply temporal slot attention for object tracking.

        Args:
            inputs: [B, N, input_dim] input features
            use_temporal: Whether to use temporal tracking (set False for first frame)
            return_tracking_info: Return additional tracking information

        Returns:
            slots: [B, num_slots, slot_dim] object slots
            attn: [B, num_slots, N] attention maps
            tracking_info: Dict with existence, occlusion, velocities, etc.
        """
        batch_size, num_inputs, input_dim = inputs.size()
        device = inputs.device

        # Initialize or check memory
        if self.slot_memory is None or self.slot_memory.size(0) != batch_size:
            self.reset_memory(batch_size, device)

        # Initialize slots
        if use_temporal and self.slot_memory is not None and self.slot_memory.abs().sum() > 0:
            # Use previous slots as initialization (with decay)
            predicted_slots = self.predict_occluded_slots(
                self.slot_memory, self.slot_velocities
            )
            # Add some noise for exploration
            noise = torch.randn_like(predicted_slots) * 0.1
            slots = predicted_slots + noise
        else:
            # Standard random initialization
            mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
            sigma = torch.exp(self.slots_log_sigma).expand(batch_size, self.num_slots, -1)
            slots = mu + sigma * torch.randn_like(mu)

        # Normalize inputs
        inputs_norm = self.norm_inputs(inputs)

        # Compute keys and values
        k = self.project_k(inputs_norm)
        v = self.project_v(inputs_norm)

        # Iterative attention refinement
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Compute attention
            q = self.project_q(slots)
            scale = self.slot_dim ** -0.5
            attn_logits = torch.einsum('bsd,bnd->bsn', q, k) * scale
            attn = F.softmax(attn_logits, dim=1)

            # Weighted mean
            attn_sum = attn.sum(dim=-1, keepdim=True) + self.epsilon
            updates = torch.einsum('bsn,bnd->bsd', attn / attn_sum, v)

            # Update slots
            slots = slots_prev + updates
            slots = slots + self.mlp(self.norm_mlp(slots))

        # Temporal integration
        if use_temporal and self.slot_memory is not None and self.slot_memory.abs().sum() > 0:
            # Match slots across frames
            match_probs, assignment = self.compute_slot_matching(slots, self.slot_memory)

            # Update velocities
            velocities = self.update_velocities(slots, self.slot_memory, assignment)

            # GRU-based temporal integration
            slots_flat = slots.view(batch_size * self.num_slots, self.slot_dim)
            memory_flat = self.slot_memory.view(batch_size * self.num_slots, self.slot_dim)
            slots_temporal = self.temporal_gru(slots_flat, memory_flat)
            slots = slots_temporal.view(batch_size, self.num_slots, self.slot_dim)

            # Update velocity estimates with decay
            self.slot_velocities = self.memory_decay * self.slot_velocities + (1 - self.memory_decay) * velocities
        else:
            assignment = torch.arange(self.num_slots, device=device).unsqueeze(0).expand(batch_size, -1)
            match_probs = torch.eye(self.num_slots, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        # Update memory
        self.slot_memory = slots.detach()
        self.object_ages = self.object_ages + 1

        # Predict existence and occlusion
        existence = self.existence_predictor(slots).squeeze(-1)
        occlusion = self.occlusion_predictor(slots).squeeze(-1)

        tracking_info = {
            'existence': existence,
            'occlusion': occlusion,
            'velocities': self.slot_velocities,
            'assignment': assignment,
            'match_probs': match_probs,
            'object_ages': self.object_ages,
        }

        if return_tracking_info:
            return slots, attn, tracking_info

        return slots, attn


class SceneGraphBuilder(nn.Module):
    """Build complete scene graphs from observations with spatial relations and bounding boxes."""

    def __init__(
        self,
        obs_dim: int,
        object_dim: int = 128,
        relation_dim: int = 64,
        max_objects: int = 10,
        num_object_types: int = 20,
        use_spatial_relations: bool = True,
        use_temporal_tracking: bool = False,
    ):
        super().__init__()
        self.max_objects = max_objects
        self.object_dim = object_dim
        self.relation_dim = relation_dim
        self.use_spatial_relations = use_spatial_relations
        self.use_temporal_tracking = use_temporal_tracking

        self.object_detector = ObjectDetector(
            obs_dim=obs_dim,
            object_dim=object_dim,
            max_objects=max_objects,
            num_object_types=num_object_types
        )

        self.relation_network = RelationNetwork(
            object_dim=object_dim,
            relation_dim=relation_dim
        )

        # Spatial relation network for explicit spatial relations
        if use_spatial_relations:
            self.spatial_relation_network = SpatialRelationNetwork(
                hidden_dim=relation_dim
            )

        # Temporal slot attention for object permanence
        if use_temporal_tracking:
            self.temporal_tracker = TemporalSlotAttention(
                num_slots=max_objects,
                slot_dim=object_dim,
                input_dim=obs_dim
            )

    def forward(
        self,
        obs: torch.Tensor,
        return_batch: bool = False
    ) -> SceneGraph:
        """Parse observation into scene graph.

        Args:
            obs: [B, obs_dim] or [B, H, W, C]
            return_batch: If True, return batched scene graph data

        Returns:
            scene_graph: SceneGraph object with all fields populated
        """
        batch_size = obs.size(0)

        # Detect objects with bounding boxes
        objects, object_types, bounding_boxes = self.object_detector(obs)

        # Detect semantic relations
        relations, relation_types, existence = self.relation_network(objects)

        # Compute spatial relations from bounding boxes
        spatial_relations = None
        if self.use_spatial_relations:
            soft_spatial, spatial_features = self.spatial_relation_network(bounding_boxes)
            spatial_relations = soft_spatial

        # Build scene graph
        if return_batch:
            scene_graph = SceneGraph(
                objects=objects,
                relations=relations,
                object_types=torch.argmax(object_types, dim=-1),
                relation_types=torch.argmax(relation_types, dim=-1),
                bounding_boxes=bounding_boxes
            )
        else:
            # Return first batch item for backward compatibility
            scene_graph = SceneGraph(
                objects=objects[0],
                relations=relations[0],
                object_types=torch.argmax(object_types[0], dim=-1),
                relation_types=torch.argmax(relation_types[0], dim=-1),
                bounding_boxes=bounding_boxes[0]
            )

        return scene_graph

    def forward_temporal(
        self,
        obs: torch.Tensor,
        use_temporal: bool = True
    ) -> Tuple[SceneGraph, Dict]:
        """Parse observation with temporal tracking for object permanence.

        Args:
            obs: [B, obs_dim] or [B, H, W, C]
            use_temporal: Whether to use temporal tracking

        Returns:
            scene_graph: SceneGraph object
            tracking_info: Dictionary with tracking information
        """
        if not self.use_temporal_tracking:
            raise ValueError("Temporal tracking not enabled. Set use_temporal_tracking=True")

        batch_size = obs.size(0)

        # Flatten spatial dims if needed
        if obs.dim() == 4:
            obs_flat = obs.flatten(1, 2)
        elif obs.dim() == 2:
            obs_flat = obs.unsqueeze(1)
        else:
            obs_flat = obs

        # Get temporal slots with tracking
        slots, attn, tracking_info = self.temporal_tracker(
            obs_flat, use_temporal=use_temporal, return_tracking_info=True
        )

        # Decode object properties from tracked slots
        objects = self.object_detector.property_decoder(slots)
        object_types = self.object_detector.type_classifier(slots)
        bounding_boxes = self.object_detector.bbox_predictor(slots)

        # Detect relations
        relations, relation_types, existence = self.relation_network(objects)

        # Build scene graph
        scene_graph = SceneGraph(
            objects=objects[0],
            relations=relations[0],
            object_types=torch.argmax(object_types[0], dim=-1),
            relation_types=torch.argmax(relation_types[0], dim=-1),
            bounding_boxes=bounding_boxes[0]
        )

        return scene_graph, tracking_info

    def encode_scene(self, scene_graph: SceneGraph) -> torch.Tensor:
        """Encode scene graph back to vector representation."""
        # Pool objects
        object_features = scene_graph.objects.mean(dim=0)

        # Pool relations
        relation_features = scene_graph.relations.mean(dim=(0, 1))

        # Concatenate
        scene_encoding = torch.cat([object_features, relation_features], dim=-1)

        return scene_encoding

    def reset_temporal_tracking(self, batch_size: int, device: torch.device) -> None:
        """Reset temporal tracking memory for new sequences."""
        if self.use_temporal_tracking:
            self.temporal_tracker.reset_memory(batch_size, device)


class PhysicsEngine(nn.Module):
    """Simple learned physics engine for object dynamics."""
    
    def __init__(
        self,
        object_dim: int,
        action_dim: int,
        hidden_size: int = 256,
    ):
        super().__init__()
        
        # Object dynamics predictor
        self.dynamics_net = nn.Sequential(
            nn.Linear(object_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, object_dim)
        )
        
        # Physics constraints
        self.gravity = nn.Parameter(torch.tensor([0.0, -9.8, 0.0]))
        self.friction = nn.Parameter(torch.tensor(0.1))
        
    def forward(
        self,
        objects: torch.Tensor,
        action: torch.Tensor,
        dt: float = 0.1
    ) -> torch.Tensor:
        """Predict next object states.
        
        Args:
            objects: [B, N, object_dim]
            action: [B, action_dim]
            dt: Time step
            
        Returns:
            next_objects: [B, N, object_dim]
        """
        batch_size, num_objects, object_dim = objects.size()
        
        # Expand action for each object
        action_expanded = action.unsqueeze(1).expand(-1, num_objects, -1)
        
        # Concatenate object and action
        inputs = torch.cat([objects, action_expanded], dim=-1)
        inputs = inputs.view(batch_size * num_objects, -1)
        
        # Predict dynamics
        delta_objects = self.dynamics_net(inputs)
        delta_objects = delta_objects.view(batch_size, num_objects, object_dim)
        
        # Apply physics constraints (gravity, etc.)
        # Assume first 3 dims represent position
        if object_dim >= 3:
            gravity_effect = self.gravity * dt
            delta_objects[:, :, :3] = delta_objects[:, :, :3] + gravity_effect
        
        # Update objects
        next_objects = objects + delta_objects * dt
        
        return next_objects
    
    def apply_collision(
        self,
        objects: torch.Tensor,
        relations: torch.Tensor
    ) -> torch.Tensor:
        """Apply collision constraints."""
        # Simplified collision: if objects overlap, push apart
        # This is a placeholder for more sophisticated collision detection
        return objects


class GroundedWorldModel(nn.Module):
    """World model grounded in object-centric representation."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        object_dim: int = 128,
        relation_dim: int = 64,
        max_objects: int = 10,
    ):
        super().__init__()
        
        self.scene_parser = SceneGraphBuilder(
            obs_dim=obs_dim,
            object_dim=object_dim,
            relation_dim=relation_dim,
            max_objects=max_objects
        )
        
        self.physics_engine = PhysicsEngine(
            object_dim=object_dim,
            action_dim=action_dim
        )
        
        # Renderer: convert objects back to observation space
        self.renderer = nn.Sequential(
            nn.Linear(object_dim * max_objects + relation_dim * max_objects * max_objects, obs_dim * 2),
            nn.ReLU(),
            nn.Linear(obs_dim * 2, obs_dim)
        )
        
        self.max_objects = max_objects
        self.relation_dim = relation_dim
        
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """Predict next observation through object-centric reasoning.
        
        Args:
            obs:[B, obs_dim]
            action: [B, action_dim]
            
        Returns:
            next_obs: [B, obs_dim]
        """
        # Parse scene into objects
        scene_graph = self.scene_parser(obs)
        
        # Add batch dimension if needed
        objects = scene_graph.objects.unsqueeze(0) if scene_graph.objects.dim() == 2 else scene_graph.objects
        relations = scene_graph.relations
        
        # Predict object dynamics
        next_objects = self.physics_engine(objects, action)
        
        # Apply collision constraints
        next_objects = self.physics_engine.apply_collision(next_objects, relations)
        
        # Render back to observation space
        # Flatten objects and relations
        batch_size = next_objects.size(0)
        objects_flat = next_objects.view(batch_size, -1)
        
        # Pad if necessary
        expected_size = self.max_objects * next_objects.size(-1)
        if objects_flat.size(-1) < expected_size:
            padding = torch.zeros(batch_size, expected_size - objects_flat.size(-1), device=objects_flat.device)
            objects_flat = torch.cat([objects_flat, padding], dim=-1)
        
        relations_flat = relations.view(batch_size, -1) if relations.dim() == 4 else relations.unsqueeze(0).view(batch_size, -1)
        expected_rel_size = self.max_objects * self.max_objects * self.relation_dim
        if relations_flat.size(-1) < expected_rel_size:
            padding = torch.zeros(batch_size, expected_rel_size - relations_flat.size(-1), device=relations_flat.device)
            relations_flat = torch.cat([relations_flat, padding], dim=-1)
        
        scene_flat = torch.cat([objects_flat[:, :expected_size], relations_flat[:, :expected_rel_size]], dim=-1)
        
        next_obs = self.renderer(scene_flat)
        
        return next_obs
