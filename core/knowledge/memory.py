"""Knowledge representation and retrieval for AGI."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class KnowledgeGraph(nn.Module):
    """Graph-structured knowledge base with entities and relations."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        self.score_net = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def score_triple(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor
    ) -> torch.Tensor:
        """Score (head, relation, tail) triple."""
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)
        
        combined = torch.cat([h, r, t], dim=-1)
        return self.score_net(combined).squeeze(-1)

    def query(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query knowledge graph for top-k tail entities."""
        batch_size = head.size(0)
        all_tails = torch.arange(self.num_entities, device=head.device)
        
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


class SemanticMemory(nn.Module):
    """Long-term semantic memory with factual knowledge."""

    def __init__(
        self,
        capacity: int,
        key_dim: int,
        value_dim: int,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.capacity = capacity
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        self.keys = nn.Parameter(torch.randn(capacity, key_dim))
        self.values = nn.Parameter(torch.randn(capacity, value_dim))
        
        self.query_projection = nn.Linear(key_dim, key_dim)
        self.attention = nn.MultiheadAttention(key_dim, num_heads, batch_first=True)
        
        nn.init.xavier_uniform_(self.keys)
        nn.init.xavier_uniform_(self.values)

    def retrieve(
        self,
        query: torch.Tensor,
        k: int = 5,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve top-k relevant memories."""
        batch_size = query.size(0)
        
        q = self.query_projection(query)
        
        scores = torch.matmul(q, self.keys.T) / temperature
        
        top_scores, top_indices = torch.topk(scores, k, dim=-1)
        
        retrieved_values = self.values[top_indices]
        attention_weights = F.softmax(top_scores, dim=-1)
        
        return retrieved_values, attention_weights

    def update(self, key: torch.Tensor, value: torch.Tensor, index: int) -> None:
        """Update memory slot."""
        if index < 0 or index >= self.capacity:
            raise ValueError(f"Index {index} out of range [0, {self.capacity})")
        
        with torch.no_grad():
            self.keys[index] = key
            self.values[index] = value


class EpisodicMemory(nn.Module):
    """Episodic memory buffer for storing event sequences."""

    def __init__(
        self,
        capacity: int,
        sequence_length: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        self.episodes = nn.Parameter(
            torch.zeros(capacity, sequence_length, hidden_size),
            requires_grad=False
        )
        self.episode_embeddings = nn.Parameter(
            torch.randn(capacity, hidden_size)
        )
        self.write_pointer = 0
        
        self.compressor = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def add_episode(self, sequence: torch.Tensor) -> int:
        """Add new episode to memory."""
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
            
            _, (h_n, _) = self.compressor(sequence.unsqueeze(0))
            self.episode_embeddings[write_idx] = h_n.squeeze(0)
        
        self.write_pointer += 1
        return write_idx

    def retrieve_similar(
        self,
        query: torch.Tensor,
        k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve k most similar episodes."""
        # Handle both single and batched queries
        if query.dim() == 1:
            query = query.unsqueeze(0)  # [hidden_size] -> [1, hidden_size]
        
        # query: [batch_size, hidden_size]
        # episode_embeddings: [capacity, hidden_size]
        # We need to compute similarity for each query
        batch_size = query.size(0)
        
        # Expand for broadcasting: [batch_size, capacity, hidden_size]
        query_expanded = query.unsqueeze(1)  # [batch_size, 1, hidden_size]
        embeddings_expanded = self.episode_embeddings.unsqueeze(0)  # [1, capacity, hidden_size]
        
        # Compute cosine similarity
        similarities = F.cosine_similarity(
            query_expanded,
            embeddings_expanded,
            dim=-1
        )  # [batch_size, capacity]
        
        # Get top k for each batch item
        k = min(k, self.capacity)
        top_scores, top_indices = torch.topk(similarities, k, dim=-1)
        
        # Retrieve episodes - take first batch item for now
        retrieved_episodes = self.episodes[top_indices[0]]
        return retrieved_episodes, top_scores[0]


class HierarchicalMemory(nn.Module):
    """Unified hierarchical memory system."""

    def __init__(
        self,
        working_memory_slots: int,
        semantic_capacity: int,
        episodic_capacity: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        
        self.working_memory = nn.Parameter(
            torch.zeros(working_memory_slots, hidden_size)
        )
        
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

    def forward(
        self,
        query: torch.Tensor,
        mode: str = "auto"
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Query hierarchical memory."""
        # Handle batch dimension
        if query.dim() == 1:
            query = query.unsqueeze(0)
        batch_size = query.size(0)
        
        if mode == "auto":
            routing_weights = self.memory_router(query)  # [batch, 3]
        elif mode == "working":
            routing_weights = torch.tensor([[1.0, 0.0, 0.0]], device=query.device).expand(batch_size, 3)
        elif mode == "semantic":
            routing_weights = torch.tensor([[0.0, 1.0, 0.0]], device=query.device).expand(batch_size, 3)
        elif mode == "episodic":
            routing_weights = torch.tensor([[0.0, 0.0, 1.0]], device=query.device).expand(batch_size, 3)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Get outputs from each memory type
        working_output = self.working_memory.mean(dim=0).unsqueeze(0).expand(batch_size, -1)  # [batch, hidden]
        
        semantic_values, semantic_weights = self.semantic_memory.retrieve(query, k=5)
        semantic_output = (semantic_values * semantic_weights.unsqueeze(-1)).sum(dim=1)  # [batch, hidden]
        
        episodic_sequences, episodic_scores = self.episodic_memory.retrieve_similar(query, k=3)
        # episodic_sequences: [k, seq_len, hidden]
        episodic_output = episodic_sequences.mean(dim=(0, 1)).unsqueeze(0).expand(batch_size, -1)  # [batch, hidden]
        
        # routing_weights: [batch, 3]
        # outputs: [batch, hidden]
        # Use einsum or manual multiplication
        combined_output = (
            routing_weights[:, 0:1] * working_output +
            routing_weights[:, 1:2] * semantic_output +
            routing_weights[:, 2:3] * episodic_output
        )  # [batch, hidden]
        
        info = {
            "routing_weights": routing_weights,
            "semantic_weights": semantic_weights,
            "episodic_scores": episodic_scores,
        }
        
        return combined_output, info


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
