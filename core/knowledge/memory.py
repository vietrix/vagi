"""Knowledge representation and retrieval for AGI.

Enhanced with:
1. Multiple scoring functions (TransE, RotatE, DistMult)
2. Approximate nearest neighbor for efficient queries
3. Uncertainty quantification for retrievals
4. Memory conflict resolution
5. Dynamic resizing with LRU eviction (SemanticMemory)
6. Online key update with contrastive loss (SemanticMemory)
7. Lossy compression for older memories (EpisodicMemory)
8. MC dropout/ensemble uncertainty quantification
9. Recency-weighted attention-based conflict resolution (HierarchicalMemory)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, NamedTuple
from enum import Enum
from collections import OrderedDict
import math
import time

import torch
from torch import nn
from torch.nn import functional as F


class MemoryEntry(NamedTuple):
    """Entry in memory with metadata for LRU tracking."""
    key: torch.Tensor
    value: torch.Tensor
    timestamp: float
    access_count: int


class ScoringFunction(Enum):
    """Knowledge graph scoring functions."""
    MLP = "mlp"          # Learned MLP scorer
    TRANSE = "transe"    # Translation-based
    DISTMULT = "distmult"  # Bilinear diagonal
    ROTATE = "rotate"    # Rotation in complex space


class KnowledgeGraph(nn.Module):
    """Graph-structured knowledge base with multiple scoring functions.

    Supports:
    - Multiple scoring methods (TransE, RotatE, DistMult, MLP)
    - Efficient approximate queries using locality-sensitive hashing
    - Entity typing for constrained queries
    - Uncertainty quantification
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        hidden_size: int = 256,
        scoring_function: str = "distmult",
        num_entity_types: int = 10,
        use_entity_types: bool = True,
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.scoring_fn = ScoringFunction(scoring_function)
        self.use_entity_types = use_entity_types

        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Entity type embeddings
        if use_entity_types:
            self.entity_types = nn.Embedding(num_entities, num_entity_types)
            self.type_embeddings = nn.Embedding(num_entity_types, embedding_dim // 4)

        # Scoring networks based on method
        if self.scoring_fn == ScoringFunction.MLP:
            self.score_net = nn.Sequential(
                nn.Linear(embedding_dim * 3, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )

        # For RotatE, we use complex embeddings (split real/imag)
        if self.scoring_fn == ScoringFunction.ROTATE:
            self.phase_relation = nn.Embedding(num_relations, embedding_dim)

        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )

        # Locality-sensitive hashing for approximate queries
        self.num_hash_tables = 4
        self.hash_dim = 32
        self.hash_projections = nn.Parameter(
            torch.randn(self.num_hash_tables, embedding_dim, self.hash_dim)
        )

        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def score_triple(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        return_uncertainty: bool = False
    ) -> torch.Tensor:
        """Score (head, relation, tail) triple using configured scoring function."""
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)

        if self.scoring_fn == ScoringFunction.MLP:
            combined = torch.cat([h, r, t], dim=-1)
            score = self.score_net(combined).squeeze(-1)

        elif self.scoring_fn == ScoringFunction.TRANSE:
            # TransE: h + r ≈ t
            score = -torch.norm(h + r - t, p=2, dim=-1)

        elif self.scoring_fn == ScoringFunction.DISTMULT:
            # DistMult: <h, r, t> (element-wise product sum)
            score = (h * r * t).sum(dim=-1)

        elif self.scoring_fn == ScoringFunction.ROTATE:
            # RotatE: rotation in complex space
            phase = self.phase_relation(relation)
            # Split into real and imaginary parts
            re_h, im_h = h.chunk(2, dim=-1)
            re_t, im_t = t.chunk(2, dim=-1)
            re_r = torch.cos(phase)
            im_r = torch.sin(phase)

            # Rotation: (re_h + i*im_h) * (re_r + i*im_r) = re_t + i*im_t
            re_score = re_h * re_r - im_h * im_r - re_t
            im_score = re_h * im_r + im_h * re_r - im_t

            score = -torch.norm(torch.cat([re_score, im_score], dim=-1), p=2, dim=-1)

        else:
            raise ValueError(f"Unknown scoring function: {self.scoring_fn}")

        if return_uncertainty:
            uncertainty = self.uncertainty_head(torch.cat([h, t], dim=-1)).squeeze(-1)
            return score, uncertainty

        return score

    def _compute_hash(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute LSH hash codes for embeddings."""
        # embeddings: [N, embedding_dim]
        # hash_projections: [num_tables, embedding_dim, hash_dim]
        # Output: [N, num_tables, hash_dim]
        projected = torch.einsum("nd,thd->nth", embeddings, self.hash_projections)
        return (projected > 0).float()

    def query(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        k: int = 10,
        use_approximate: bool = True,
        candidate_ratio: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query knowledge graph for top-k tail entities.

        Args:
            head: Head entity indices [batch_size]
            relation: Relation indices [batch_size]
            k: Number of results to return
            use_approximate: Use LSH for approximate but faster queries
            candidate_ratio: Fraction of entities to consider in approximate mode
        """
        batch_size = head.size(0)
        device = head.device

        if use_approximate and self.num_entities > 1000:
            # Approximate query using LSH
            h = self.entity_embeddings(head)
            r = self.relation_embeddings(relation)

            # Predict target embedding
            if self.scoring_fn == ScoringFunction.TRANSE:
                target_embed = h + r
            else:
                target_embed = h * r

            # Hash the target
            target_hash = self._compute_hash(target_embed)

            # Hash all entities
            all_embeds = self.entity_embeddings.weight
            all_hashes = self._compute_hash(all_embeds)

            # Find candidates with matching hashes
            num_candidates = max(k * 10, int(self.num_entities * candidate_ratio))

            # Hamming distance between target and all entities
            # Sum over hash tables and hash dimensions
            hash_dist = (target_hash.unsqueeze(1) != all_hashes.unsqueeze(0)).float().sum(dim=(2, 3))

            # Get top candidates with smallest hash distance
            _, candidate_indices = torch.topk(hash_dist, num_candidates, dim=1, largest=False)

            # Score only candidates
            scores = torch.zeros(batch_size, num_candidates, device=device)
            for i in range(batch_size):
                cand = candidate_indices[i]
                scores[i] = self.score_triple(
                    head[i].expand(num_candidates),
                    relation[i].expand(num_candidates),
                    cand
                )

            # Get top-k from candidates
            top_scores, top_local_indices = torch.topk(scores, k, dim=1)
            top_indices = torch.gather(candidate_indices, 1, top_local_indices)

        else:
            # Exact query (original implementation)
            all_tails = torch.arange(self.num_entities, device=device)

            head_expanded = head.unsqueeze(1).expand(batch_size, self.num_entities)
            relation_expanded = relation.unsqueeze(1).expand(batch_size, self.num_entities)
            tail_expanded = all_tails.unsqueeze(0).expand(batch_size, self.num_entities)

            scores = self.score_triple(
                head_expanded.reshape(-1),
                relation_expanded.reshape(-1),
                tail_expanded.reshape(-1)
            ).view(batch_size, self.num_entities)

            top_scores, top_indices = torch.topk(scores, k, dim=1)

        return top_indices, top_scores

    def get_entity_type(self, entity: torch.Tensor) -> torch.Tensor:
        """Get entity type distribution."""
        if self.use_entity_types:
            return F.softmax(self.entity_types(entity), dim=-1)
        return None


class SemanticMemory(nn.Module):
    """Long-term semantic memory with factual knowledge.

    Enhanced with:
    - Dynamic resizing with LRU eviction (4.4)
    - Online key update with contrastive loss (4.5)
    - MC dropout uncertainty quantification (4.7)
    """

    def __init__(
        self,
        capacity: int,
        key_dim: int,
        value_dim: int,
        num_heads: int = 4,
        max_capacity: int = None,
        growth_factor: float = 1.5,
        contrastive_margin: float = 0.5,
        contrastive_lr: float = 0.01,
        num_mc_samples: int = 10,
        mc_dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.capacity = capacity
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.max_capacity = max_capacity or capacity * 4  # Default max 4x initial
        self.growth_factor = growth_factor
        self.contrastive_margin = contrastive_margin
        self.contrastive_lr = contrastive_lr
        self.num_mc_samples = num_mc_samples
        self.mc_dropout_rate = mc_dropout_rate

        # Main storage
        self.keys = nn.Parameter(torch.randn(capacity, key_dim))
        self.values = nn.Parameter(torch.randn(capacity, value_dim))

        # LRU tracking (4.4)
        self.register_buffer('access_timestamps', torch.zeros(capacity))
        self.register_buffer('access_counts', torch.zeros(capacity, dtype=torch.long))
        self.register_buffer('valid_mask', torch.zeros(capacity, dtype=torch.bool))
        self.current_size = 0

        # Query projection with dropout for MC uncertainty (4.7)
        self.query_projection = nn.Sequential(
            nn.Linear(key_dim, key_dim),
            nn.Dropout(mc_dropout_rate),
        )
        self.attention = nn.MultiheadAttention(key_dim, num_heads, batch_first=True, dropout=mc_dropout_rate)

        # Contrastive key update network (4.5)
        self.key_update_net = nn.Sequential(
            nn.Linear(key_dim * 2, key_dim),
            nn.ReLU(),
            nn.Linear(key_dim, key_dim),
        )

        nn.init.xavier_uniform_(self.keys)
        nn.init.xavier_uniform_(self.values)

    def _get_lru_indices(self, n: int) -> torch.Tensor:
        """Get n least recently used indices for eviction (4.4)."""
        # Combine recency and frequency for LRU-K policy
        # Lower score = more likely to evict
        recency_score = self.access_timestamps.clone()
        frequency_score = self.access_counts.float() / (self.access_counts.max() + 1e-8)

        # Combined score: higher = more important, keep
        combined_score = recency_score * 0.7 + frequency_score * 0.3

        # Only consider valid entries
        combined_score[~self.valid_mask] = float('inf')  # Don't evict empty slots

        # Get indices with lowest scores
        _, indices = torch.topk(combined_score, n, largest=False)
        return indices

    def _resize_memory(self, new_capacity: int) -> None:
        """Dynamically resize memory capacity (4.4)."""
        if new_capacity > self.max_capacity:
            new_capacity = self.max_capacity
        if new_capacity <= self.capacity:
            return

        device = self.keys.device

        # Create new larger tensors
        new_keys = torch.randn(new_capacity, self.key_dim, device=device)
        new_values = torch.randn(new_capacity, self.value_dim, device=device)
        new_timestamps = torch.zeros(new_capacity, device=device)
        new_counts = torch.zeros(new_capacity, dtype=torch.long, device=device)
        new_valid = torch.zeros(new_capacity, dtype=torch.bool, device=device)

        # Copy existing data
        new_keys[:self.capacity] = self.keys.data
        new_values[:self.capacity] = self.values.data
        new_timestamps[:self.capacity] = self.access_timestamps
        new_counts[:self.capacity] = self.access_counts
        new_valid[:self.capacity] = self.valid_mask

        # Initialize new slots
        nn.init.xavier_uniform_(new_keys[self.capacity:])
        nn.init.xavier_uniform_(new_values[self.capacity:])

        # Update parameters
        self.keys = nn.Parameter(new_keys)
        self.values = nn.Parameter(new_values)
        self.register_buffer('access_timestamps', new_timestamps)
        self.register_buffer('access_counts', new_counts)
        self.register_buffer('valid_mask', new_valid)
        self.capacity = new_capacity

    def _compute_contrastive_loss(
        self,
        query: torch.Tensor,
        positive_indices: torch.Tensor,
        negative_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss for key updates (4.5)."""
        # query: [batch, key_dim]
        # positive_indices, negative_indices: [batch, k]

        positive_keys = self.keys[positive_indices]  # [batch, k, key_dim]
        negative_keys = self.keys[negative_indices]  # [batch, k, key_dim]

        query_expanded = query.unsqueeze(1)  # [batch, 1, key_dim]

        # Compute distances
        pos_dist = torch.norm(query_expanded - positive_keys, p=2, dim=-1)  # [batch, k]
        neg_dist = torch.norm(query_expanded - negative_keys, p=2, dim=-1)  # [batch, k]

        # Triplet margin loss
        loss = F.relu(pos_dist.mean(dim=1) - neg_dist.mean(dim=1) + self.contrastive_margin)
        return loss.mean()

    def _update_keys_online(
        self,
        query: torch.Tensor,
        retrieved_indices: torch.Tensor,
        scores: torch.Tensor
    ) -> None:
        """Online key update with contrastive learning (4.5)."""
        if not self.training:
            return

        batch_size = query.size(0)
        k = retrieved_indices.size(1)

        # Top retrieved are positives (assuming good retrieval feedback)
        positive_indices = retrieved_indices[:, :max(1, k//2)]

        # Sample negatives from low-scoring entries
        num_neg = max(1, k//2)
        all_indices = torch.arange(self.capacity, device=query.device)
        valid_indices = all_indices[self.valid_mask]

        if len(valid_indices) > num_neg:
            # Sample random negatives
            neg_perm = torch.randperm(len(valid_indices), device=query.device)[:num_neg]
            negative_indices = valid_indices[neg_perm].unsqueeze(0).expand(batch_size, -1)
        else:
            negative_indices = retrieved_indices[:, -num_neg:]

        # Compute contrastive loss
        loss = self._compute_contrastive_loss(query, positive_indices, negative_indices)

        # Update keys using gradient-free approach (for inference efficiency)
        with torch.no_grad():
            # Move positive keys closer to query
            for i in range(batch_size):
                for idx in positive_indices[i]:
                    update = self.contrastive_lr * (query[i] - self.keys[idx])
                    self.keys.data[idx] += update

    def retrieve_with_uncertainty(
        self,
        query: torch.Tensor,
        k: int = 5,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve with MC dropout uncertainty estimation (4.7)."""
        batch_size = query.size(0)
        device = query.device

        # MC dropout samples
        all_scores = []
        was_training = self.training
        self.train()  # Enable dropout

        for _ in range(self.num_mc_samples):
            q = self.query_projection(query)
            scores = torch.matmul(q, self.keys.T) / temperature
            # Mask invalid entries
            scores[:, ~self.valid_mask] = float('-inf')
            all_scores.append(scores)

        if not was_training:
            self.eval()

        # Stack: [num_samples, batch, capacity]
        all_scores = torch.stack(all_scores, dim=0)

        # Mean and variance across MC samples
        mean_scores = all_scores.mean(dim=0)  # [batch, capacity]
        score_variance = all_scores.var(dim=0)  # [batch, capacity]

        # Get top-k based on mean scores
        top_scores, top_indices = torch.topk(mean_scores, k, dim=-1)

        # Gather uncertainties for top-k
        uncertainties = torch.gather(score_variance, 1, top_indices)  # [batch, k]

        retrieved_values = self.values[top_indices]
        attention_weights = F.softmax(top_scores, dim=-1)

        return retrieved_values, attention_weights, uncertainties

    def retrieve(
        self,
        query: torch.Tensor,
        k: int = 5,
        temperature: float = 1.0,
        return_uncertainty: bool = False,
        update_keys: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve top-k relevant memories.

        Enhanced with:
        - LRU access tracking (4.4)
        - Online key updates (4.5)
        - Optional uncertainty quantification (4.7)
        """
        batch_size = query.size(0)
        current_time = time.time()

        if return_uncertainty:
            retrieved_values, attention_weights, uncertainties = self.retrieve_with_uncertainty(
                query, k, temperature
            )
            # Update access tracking
            with torch.no_grad():
                for i in range(batch_size):
                    _, top_indices = torch.topk(
                        torch.matmul(self.query_projection[0](query[i:i+1]), self.keys.T).squeeze(0),
                        k, dim=-1
                    )
                    self.access_timestamps[top_indices] = current_time
                    self.access_counts[top_indices] += 1
            return retrieved_values, attention_weights, uncertainties

        q = self.query_projection(query)

        scores = torch.matmul(q, self.keys.T) / temperature
        # Mask invalid entries
        scores[:, ~self.valid_mask] = float('-inf')

        top_scores, top_indices = torch.topk(scores, min(k, self.current_size or k), dim=-1)

        # Update LRU tracking (4.4)
        with torch.no_grad():
            unique_indices = top_indices.unique()
            self.access_timestamps[unique_indices] = current_time
            self.access_counts[unique_indices] += 1

        # Online key update (4.5)
        if update_keys and self.training:
            self._update_keys_online(query, top_indices, top_scores)

        retrieved_values = self.values[top_indices]
        attention_weights = F.softmax(top_scores, dim=-1)

        return retrieved_values, attention_weights

    def update(self, key: torch.Tensor, value: torch.Tensor, index: int = None) -> int:
        """Update memory slot with LRU eviction support (4.4).

        If index is None, finds an empty slot or evicts LRU entry.
        Returns the index where the entry was stored.
        """
        device = key.device
        current_time = time.time()

        if index is not None:
            # Explicit index update
            if index < 0 or index >= self.capacity:
                raise ValueError(f"Index {index} out of range [0, {self.capacity})")

            with torch.no_grad():
                self.keys.data[index] = key
                self.values.data[index] = value
                self.access_timestamps[index] = current_time
                self.access_counts[index] = 1
                self.valid_mask[index] = True
            return index

        # Find slot: empty or LRU eviction
        empty_slots = (~self.valid_mask).nonzero(as_tuple=True)[0]

        if len(empty_slots) > 0:
            # Use first empty slot
            target_idx = empty_slots[0].item()
        else:
            # Check if we can grow
            if self.capacity < self.max_capacity:
                new_capacity = min(int(self.capacity * self.growth_factor), self.max_capacity)
                self._resize_memory(new_capacity)
                # Now there's space
                empty_slots = (~self.valid_mask).nonzero(as_tuple=True)[0]
                target_idx = empty_slots[0].item()
            else:
                # Evict LRU entry
                lru_indices = self._get_lru_indices(1)
                target_idx = lru_indices[0].item()

        with torch.no_grad():
            self.keys.data[target_idx] = key
            self.values.data[target_idx] = value
            self.access_timestamps[target_idx] = current_time
            self.access_counts[target_idx] = 1
            self.valid_mask[target_idx] = True

        self.current_size = min(self.current_size + 1, self.capacity)
        return target_idx

    def evict(self, n: int = 1) -> List[int]:
        """Explicitly evict n least recently used entries (4.4)."""
        if n <= 0:
            return []

        n = min(n, self.current_size)
        lru_indices = self._get_lru_indices(n)

        evicted = []
        with torch.no_grad():
            for idx in lru_indices:
                idx_item = idx.item()
                if self.valid_mask[idx_item]:
                    self.valid_mask[idx_item] = False
                    self.current_size -= 1
                    evicted.append(idx_item)

        return evicted


class EpisodicMemory(nn.Module):
    """Episodic memory buffer for storing event sequences.

    Enhanced with:
    - Lossy compression for older memories (4.6)
    - MC dropout uncertainty quantification (4.7)
    """

    def __init__(
        self,
        capacity: int,
        sequence_length: int,
        hidden_size: int,
        compression_threshold: int = None,
        compression_ratio: float = 0.5,
        summary_hidden_size: int = None,
        num_mc_samples: int = 10,
        mc_dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.compression_threshold = compression_threshold or capacity // 2
        self.compression_ratio = compression_ratio
        self.summary_hidden_size = summary_hidden_size or hidden_size
        self.num_mc_samples = num_mc_samples
        self.mc_dropout_rate = mc_dropout_rate

        # Full episode storage
        self.episodes = nn.Parameter(
            torch.zeros(capacity, sequence_length, hidden_size),
            requires_grad=False
        )
        self.episode_embeddings = nn.Parameter(
            torch.randn(capacity, hidden_size)
        )
        self.write_pointer = 0

        # Compression tracking (4.6)
        self.register_buffer('is_compressed', torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer('episode_timestamps', torch.zeros(capacity))
        self.register_buffer('compression_levels', torch.zeros(capacity, dtype=torch.long))

        # Compressed summaries storage (4.6)
        compressed_seq_len = max(1, int(sequence_length * compression_ratio))
        self.compressed_episodes = nn.Parameter(
            torch.zeros(capacity, compressed_seq_len, hidden_size),
            requires_grad=False
        )
        self.compressed_seq_len = compressed_seq_len

        # Compressor LSTM
        self.compressor = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Summarization network for lossy compression (4.6)
        self.summarizer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(mc_dropout_rate),
            nn.Linear(hidden_size, hidden_size),
        )
        self.summary_attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, batch_first=True, dropout=mc_dropout_rate
        )

        # Importance scoring for selective compression (4.6)
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def _compress_episode(self, episode: torch.Tensor) -> torch.Tensor:
        """Compress episode sequence using attention-based summarization (4.6).

        Args:
            episode: [seq_len, hidden_size] full episode

        Returns:
            compressed: [compressed_seq_len, hidden_size] summarized episode
        """
        episode = episode.unsqueeze(0)  # [1, seq_len, hidden]

        # Score importance of each timestep
        importance = self.importance_scorer(episode).squeeze(-1)  # [1, seq_len]

        # Select top important frames
        num_keep = self.compressed_seq_len
        _, top_indices = torch.topk(importance, num_keep, dim=-1)
        top_indices = top_indices.sort(dim=-1)[0]  # Keep temporal order

        # Gather important frames
        selected = torch.gather(
            episode,
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        )  # [1, compressed_seq_len, hidden]

        # Apply summarization to smooth/blend information
        summarized, _ = self.summary_attention(selected, episode, episode)

        # Additional summarization pass
        compressed = self.summarizer(summarized)

        return compressed.squeeze(0)  # [compressed_seq_len, hidden]

    def _trigger_compression(self) -> None:
        """Compress older episodes when threshold is reached (4.6)."""
        current_time = time.time()
        num_episodes = min(self.write_pointer, self.capacity)

        if num_episodes < self.compression_threshold:
            return

        # Find uncompressed episodes
        uncompressed_mask = ~self.is_compressed
        uncompressed_indices = uncompressed_mask.nonzero(as_tuple=True)[0]

        if len(uncompressed_indices) == 0:
            return

        # Sort by timestamp (oldest first)
        timestamps = self.episode_timestamps[uncompressed_indices]
        sorted_order = timestamps.argsort()
        sorted_indices = uncompressed_indices[sorted_order]

        # Compress oldest half of uncompressed episodes
        num_to_compress = max(1, len(sorted_indices) // 2)

        with torch.no_grad():
            for idx in sorted_indices[:num_to_compress]:
                idx_item = idx.item()
                episode = self.episodes[idx_item]

                # Compress and store
                compressed = self._compress_episode(episode)
                self.compressed_episodes.data[idx_item] = compressed
                self.is_compressed[idx_item] = True
                self.compression_levels[idx_item] += 1

    def add_episode(self, sequence: torch.Tensor) -> int:
        """Add new episode to memory with automatic compression trigger (4.6)."""
        current_time = time.time()

        if sequence.size(0) > self.sequence_length:
            sequence = sequence[:self.sequence_length]
        elif sequence.size(0) < self.sequence_length:
            padding = torch.zeros(
                self.sequence_length - sequence.size(0),
                self.hidden_size,
                device=sequence.device
            )
            sequence = torch.cat([sequence, padding], dim=0)

        write_idx = self.write_pointer % self.capacity

        with torch.no_grad():
            self.episodes[write_idx] = sequence
            self.episode_timestamps[write_idx] = current_time
            self.is_compressed[write_idx] = False  # New episodes are not compressed
            self.compression_levels[write_idx] = 0

            _, (h_n, _) = self.compressor(sequence.unsqueeze(0))
            self.episode_embeddings[write_idx] = h_n.squeeze(0)

        self.write_pointer += 1

        # Trigger compression if threshold reached
        self._trigger_compression()

        return write_idx

    def get_episode(self, idx: int, prefer_full: bool = False) -> torch.Tensor:
        """Get episode, returning compressed version for old episodes (4.6)."""
        if prefer_full or not self.is_compressed[idx]:
            return self.episodes[idx]
        else:
            return self.compressed_episodes[idx]

    def retrieve_similar_with_uncertainty(
        self,
        query: torch.Tensor,
        k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve with MC dropout uncertainty estimation (4.7)."""
        if query.dim() == 1:
            query = query.unsqueeze(0)

        batch_size = query.size(0)
        device = query.device

        # MC dropout samples
        all_similarities = []
        was_training = self.training
        self.train()

        for _ in range(self.num_mc_samples):
            # Apply dropout to embeddings via summarizer
            noisy_embeddings = self.summarizer(self.episode_embeddings)

            query_expanded = query.unsqueeze(1)
            embeddings_expanded = noisy_embeddings.unsqueeze(0)

            similarities = F.cosine_similarity(
                query_expanded, embeddings_expanded, dim=-1
            )
            all_similarities.append(similarities)

        if not was_training:
            self.eval()

        # Stack: [num_samples, batch, capacity]
        all_similarities = torch.stack(all_similarities, dim=0)

        # Mean and variance
        mean_sim = all_similarities.mean(dim=0)
        sim_variance = all_similarities.var(dim=0)

        # Get top-k
        k = min(k, self.capacity)
        top_scores, top_indices = torch.topk(mean_sim, k, dim=-1)

        # Gather uncertainties
        uncertainties = torch.gather(sim_variance, 1, top_indices)

        # Retrieve episodes (using compressed versions where applicable)
        batch_size = top_indices.size(0)
        retrieved_episodes = []
        for b in range(batch_size):
            batch_episodes = []
            for idx in top_indices[b]:
                idx_item = idx.item()
                if self.is_compressed[idx_item]:
                    # Return compressed episode, padded to full size
                    compressed = self.compressed_episodes[idx_item]
                    padding = torch.zeros(
                        self.sequence_length - self.compressed_seq_len,
                        self.hidden_size,
                        device=device
                    )
                    padded = torch.cat([compressed, padding], dim=0)
                    batch_episodes.append(padded)
                else:
                    batch_episodes.append(self.episodes[idx_item])
            retrieved_episodes.append(torch.stack(batch_episodes, dim=0))

        retrieved_episodes = torch.stack(retrieved_episodes, dim=0)

        return retrieved_episodes, top_scores, uncertainties

    def retrieve_similar(
        self,
        query: torch.Tensor,
        k: int = 3,
        return_uncertainty: bool = False,
        use_compressed: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve k most similar episodes.

        Enhanced with:
        - Lossy compression support (4.6)
        - Optional uncertainty quantification (4.7)
        """
        if return_uncertainty:
            return self.retrieve_similar_with_uncertainty(query, k)

        # Handle both single and batched queries
        if query.dim() == 1:
            query = query.unsqueeze(0)

        batch_size = query.size(0)
        device = query.device

        # Expand for broadcasting
        query_expanded = query.unsqueeze(1)
        embeddings_expanded = self.episode_embeddings.unsqueeze(0)

        # Compute cosine similarity
        similarities = F.cosine_similarity(
            query_expanded, embeddings_expanded, dim=-1
        )

        # Get top k
        k = min(k, self.capacity)
        top_scores, top_indices = torch.topk(similarities, k, dim=-1)

        # Retrieve episodes
        if use_compressed:
            # Handle compressed episodes (4.6)
            retrieved_episodes = []
            for b in range(batch_size):
                batch_episodes = []
                for idx in top_indices[b]:
                    idx_item = idx.item()
                    if self.is_compressed[idx_item]:
                        compressed = self.compressed_episodes[idx_item]
                        padding = torch.zeros(
                            self.sequence_length - self.compressed_seq_len,
                            self.hidden_size,
                            device=device
                        )
                        padded = torch.cat([compressed, padding], dim=0)
                        batch_episodes.append(padded)
                    else:
                        batch_episodes.append(self.episodes[idx_item])
                retrieved_episodes.append(torch.stack(batch_episodes, dim=0))
            retrieved_episodes = torch.stack(retrieved_episodes, dim=0)
        else:
            # Original behavior: always return full episodes
            retrieved_episodes = self.episodes[top_indices.reshape(-1)]
            retrieved_episodes = retrieved_episodes.reshape(
                batch_size, k, self.sequence_length, self.hidden_size
            )

        return retrieved_episodes, top_scores

    def get_compression_stats(self) -> Dict[str, any]:
        """Get statistics about memory compression (4.6)."""
        num_episodes = min(self.write_pointer, self.capacity)
        num_compressed = self.is_compressed.sum().item()
        avg_compression_level = self.compression_levels.float().mean().item()

        return {
            "total_episodes": num_episodes,
            "compressed_episodes": num_compressed,
            "compression_ratio": num_compressed / max(1, num_episodes),
            "avg_compression_level": avg_compression_level,
            "memory_saved_ratio": (num_compressed * (1 - self.compression_ratio)) / max(1, num_episodes),
        }


class HierarchicalMemory(nn.Module):
    """Unified hierarchical memory system.

    Enhanced with:
    - Recency-weighted attention-based conflict resolution (4.8)
    - Ensemble uncertainty estimation (4.7)
    """

    def __init__(
        self,
        working_memory_slots: int,
        semantic_capacity: int,
        episodic_capacity: int,
        hidden_size: int,
        recency_decay: float = 0.95,
        conflict_threshold: float = 0.3,
        num_ensemble_heads: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.recency_decay = recency_decay
        self.conflict_threshold = conflict_threshold
        self.num_ensemble_heads = num_ensemble_heads

        # Working memory with recency tracking (4.8)
        self.working_memory = nn.Parameter(
            torch.zeros(working_memory_slots, hidden_size)
        )
        self.register_buffer('working_memory_timestamps', torch.zeros(working_memory_slots))
        self.register_buffer('working_memory_access_counts', torch.zeros(working_memory_slots, dtype=torch.long))
        self.working_memory_slots = working_memory_slots

        self.semantic_memory = SemanticMemory(
            capacity=semantic_capacity,
            key_dim=hidden_size,
            value_dim=hidden_size
        )

        self.episodic_memory = EpisodicMemory(
            capacity=episodic_capacity,
            sequence_length=16,
            hidden_size=hidden_size
        )

        self.memory_router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
            nn.Softmax(dim=-1)
        )

        # Conflict detection network (4.8)
        self.conflict_detector = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Conflict resolution attention (4.8)
        self.conflict_resolution_attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, batch_first=True
        )

        # Recency-weighted fusion network (4.8)
        self.recency_fusion = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),  # +1 for recency score
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Ensemble heads for uncertainty estimation (4.7)
        self.ensemble_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size),
            )
            for _ in range(num_ensemble_heads)
        ])

    def _compute_recency_weights(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Compute recency weights using exponential decay (4.8)."""
        current_time = time.time()
        time_deltas = current_time - timestamps

        # Exponential decay: more recent = higher weight
        # Normalize time deltas to reasonable scale (seconds -> hours)
        normalized_deltas = time_deltas / 3600.0
        recency_weights = torch.pow(
            torch.tensor(self.recency_decay, device=timestamps.device),
            normalized_deltas
        )

        return recency_weights

    def _detect_conflict(
        self,
        working_output: torch.Tensor,
        semantic_output: torch.Tensor,
        episodic_output: torch.Tensor
    ) -> torch.Tensor:
        """Detect conflicts between memory sources (4.8)."""
        # Concatenate all memory outputs
        combined = torch.cat([working_output, semantic_output, episodic_output], dim=-1)

        # Predict conflict probability
        conflict_prob = self.conflict_detector(combined)

        return conflict_prob

    def _resolve_conflict(
        self,
        working_output: torch.Tensor,
        semantic_output: torch.Tensor,
        episodic_output: torch.Tensor,
        working_recency: torch.Tensor,
        semantic_recency: torch.Tensor,
        episodic_recency: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve conflicts using recency-weighted attention (4.8)."""
        batch_size = query.size(0)
        device = query.device

        # Stack memory outputs: [batch, 3, hidden]
        memory_outputs = torch.stack([working_output, semantic_output, episodic_output], dim=1)

        # Compute recency scores for each memory type
        # Average recency per memory type
        recency_scores = torch.stack([
            working_recency.mean().expand(batch_size),
            semantic_recency.mean().expand(batch_size) if semantic_recency.numel() > 0 else torch.zeros(batch_size, device=device),
            episodic_recency.mean().expand(batch_size) if episodic_recency.numel() > 0 else torch.zeros(batch_size, device=device),
        ], dim=1)  # [batch, 3]

        # Apply recency weighting to attention
        query_expanded = query.unsqueeze(1)  # [batch, 1, hidden]

        # Attention with recency bias
        attn_scores = torch.matmul(query_expanded, memory_outputs.transpose(-2, -1))  # [batch, 1, 3]
        attn_scores = attn_scores.squeeze(1) / math.sqrt(self.hidden_size)  # [batch, 3]

        # Add recency bias
        recency_bias = recency_scores * 0.5  # Scale recency influence
        attn_scores = attn_scores + recency_bias

        # Softmax attention
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, 3]

        # Weighted combination
        resolved = (memory_outputs * attn_weights.unsqueeze(-1)).sum(dim=1)  # [batch, hidden]

        return resolved, attn_weights

    def _ensemble_uncertainty(
        self,
        working_output: torch.Tensor,
        semantic_output: torch.Tensor,
        episodic_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ensemble-based uncertainty (4.7)."""
        # Concatenate inputs
        combined = torch.cat([working_output, semantic_output, episodic_output], dim=-1)

        # Get predictions from each ensemble head
        predictions = []
        for head in self.ensemble_heads:
            pred = head(combined)
            predictions.append(pred)

        # Stack: [num_ensemble, batch, hidden]
        predictions = torch.stack(predictions, dim=0)

        # Mean prediction and uncertainty (variance)
        mean_pred = predictions.mean(dim=0)  # [batch, hidden]
        uncertainty = predictions.var(dim=0).mean(dim=-1)  # [batch]

        return mean_pred, uncertainty

    def update_working_memory(self, index: int, value: torch.Tensor) -> None:
        """Update working memory slot with timestamp (4.8)."""
        if index < 0 or index >= self.working_memory_slots:
            raise ValueError(f"Index {index} out of range")

        with torch.no_grad():
            self.working_memory.data[index] = value
            self.working_memory_timestamps[index] = time.time()
            self.working_memory_access_counts[index] += 1

    def forward(
        self,
        query: torch.Tensor,
        mode: str = "auto",
        resolve_conflicts: bool = True,
        return_uncertainty: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Query hierarchical memory.

        Enhanced with:
        - Conflict detection and resolution (4.8)
        - Ensemble uncertainty estimation (4.7)
        """
        # Handle batch dimension
        if query.dim() == 1:
            query = query.unsqueeze(0)
        batch_size = query.size(0)
        device = query.device
        current_time = time.time()

        if mode == "auto":
            routing_weights = self.memory_router(query)  # [batch, 3]
        elif mode == "working":
            routing_weights = torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(batch_size, 3)
        elif mode == "semantic":
            routing_weights = torch.tensor([[0.0, 1.0, 0.0]], device=device).expand(batch_size, 3)
        elif mode == "episodic":
            routing_weights = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(batch_size, 3)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Get outputs from each memory type with recency tracking (4.8)
        # Working memory
        working_recency = self._compute_recency_weights(self.working_memory_timestamps)
        working_attn = torch.matmul(query, self.working_memory.T)  # [batch, slots]

        # Apply recency weighting to working memory attention
        working_attn = working_attn * working_recency.unsqueeze(0)  # Recency bias
        working_attn = F.softmax(working_attn / (self.hidden_size ** 0.5), dim=-1)
        working_output = torch.matmul(working_attn, self.working_memory)  # [batch, hidden]

        # Update working memory access timestamps
        with torch.no_grad():
            # Track which slots were accessed
            top_working_indices = working_attn.argmax(dim=-1)
            for idx in top_working_indices.unique():
                self.working_memory_timestamps[idx] = current_time
                self.working_memory_access_counts[idx] += 1

        # Semantic memory with recency
        semantic_values, semantic_weights = self.semantic_memory.retrieve(query, k=5)
        semantic_output = (semantic_values * semantic_weights.unsqueeze(-1)).sum(dim=1)
        semantic_recency = self.semantic_memory.access_timestamps[
            self.semantic_memory.valid_mask
        ] if self.semantic_memory.current_size > 0 else torch.zeros(1, device=device)

        # Episodic memory with recency
        episodic_sequences, episodic_scores = self.episodic_memory.retrieve_similar(query, k=3)
        episodic_weights = F.softmax(episodic_scores, dim=-1)
        episodic_pooled = episodic_sequences.mean(dim=2)
        episodic_output = (episodic_pooled * episodic_weights.unsqueeze(-1)).sum(dim=1)

        # Get episodic recency
        num_episodes = min(self.episodic_memory.write_pointer, self.episodic_memory.capacity)
        episodic_recency = self.episodic_memory.episode_timestamps[:num_episodes] if num_episodes > 0 else torch.zeros(1, device=device)

        # Detect conflicts (4.8)
        conflict_prob = self._detect_conflict(working_output, semantic_output, episodic_output)

        # Resolve conflicts if detected and requested (4.8)
        if resolve_conflicts and (conflict_prob > self.conflict_threshold).any():
            resolved_output, resolution_weights = self._resolve_conflict(
                working_output, semantic_output, episodic_output,
                working_recency, semantic_recency, episodic_recency,
                query
            )

            # Blend resolved output with standard routing based on conflict probability
            standard_output = (
                routing_weights[:, 0:1] * working_output +
                routing_weights[:, 1:2] * semantic_output +
                routing_weights[:, 2:3] * episodic_output
            )

            # Higher conflict probability = more weight on resolved output
            combined_output = (
                conflict_prob * resolved_output +
                (1 - conflict_prob) * standard_output
            )
        else:
            # Standard routing without conflict resolution
            combined_output = (
                routing_weights[:, 0:1] * working_output +
                routing_weights[:, 1:2] * semantic_output +
                routing_weights[:, 2:3] * episodic_output
            )
            resolution_weights = routing_weights

        # Ensemble uncertainty estimation (4.7)
        if return_uncertainty:
            ensemble_output, uncertainty = self._ensemble_uncertainty(
                working_output, semantic_output, episodic_output
            )
            # Blend ensemble output
            combined_output = 0.8 * combined_output + 0.2 * ensemble_output
        else:
            uncertainty = torch.zeros(batch_size, device=device)

        # Compute memory confidence
        semantic_confidence = semantic_weights.max(dim=-1)[0].mean().item()
        episodic_confidence = episodic_scores.max(dim=-1)[0].mean().item()
        memory_confidence = (semantic_confidence + episodic_confidence) / 2

        info = {
            "routing_weights": routing_weights,
            "semantic_weights": semantic_weights,
            "episodic_scores": episodic_scores,
            "confidence": memory_confidence,
            "working_attention": working_attn,
            # New fields for enhancements
            "conflict_probability": conflict_prob,
            "resolution_weights": resolution_weights,
            "working_recency": working_recency,
            "uncertainty": uncertainty,
        }

        return combined_output, info

    def get_conflict_stats(self) -> Dict[str, any]:
        """Get statistics about memory conflicts (4.8)."""
        return {
            "working_memory_ages": (time.time() - self.working_memory_timestamps).tolist(),
            "working_memory_access_counts": self.working_memory_access_counts.tolist(),
            "semantic_compression_stats": self.episodic_memory.get_compression_stats(),
        }


class ConceptEncoder(nn.Module):
    """Encode concepts as disentangled representations."""

    def __init__(
        self,
        input_dim: int,
        concept_dim: int,
        num_concepts: int = 10,
    ) -> None:
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, concept_dim * 2),
                nn.ReLU(),
                nn.Linear(concept_dim * 2, concept_dim)
            )
            for _ in range(num_concepts)
        ])
        
        self.concept_gates = nn.Sequential(
            nn.Linear(input_dim, num_concepts),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input into disentangled concepts."""
        gates = self.concept_gates(x)
        
        concepts = []
        for encoder in self.encoders:
            concept = encoder(x)
            concepts.append(concept)
        
        concepts_tensor = torch.stack(concepts, dim=1)
        
        gated_concepts = concepts_tensor * gates.unsqueeze(-1)
        
        return gated_concepts, gates
