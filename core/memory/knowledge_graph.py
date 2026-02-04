#!/usr/bin/env python3
"""
Graph-Enhanced Memory (GraphRAG) for vAGI.

Implements a lightweight Knowledge Graph system using networkx for in-memory
graph operations. Supports SVO extraction, subgraph queries with neighbor
expansion, and integration with Transformer prompts.

Usage:
    from core.memory.knowledge_graph import KnowledgeGraphRAG

    # Initialize
    graph = KnowledgeGraphRAG(embedding_dim=768)

    # Add documents
    graph.add_document("Albert Einstein developed the theory of relativity.")
    graph.add_document("The theory of relativity changed modern physics.")

    # Query subgraph for reasoning context
    context = graph.query_subgraph("Einstein", depth=2)

    # Format for Transformer prompt
    prompt_context = graph.format_for_prompt(context)

Architecture:
    - Entity nodes with embeddings and metadata
    - Relation edges with typed connections
    - SVO extraction for automatic graph population
    - BFS traversal for subgraph retrieval
    - Semantic similarity for entity matching
"""

from __future__ import annotations

import re
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Iterator, Any
from collections import defaultdict
from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F

try:
    import networkx as nx
except ImportError:
    raise ImportError("Please install networkx: pip install networkx")

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class RelationType(Enum):
    """Common relation types for knowledge extraction."""
    # Core semantic relations
    IS_A = "is_a"                    # Taxonomy: X is a Y
    PART_OF = "part_of"              # Meronymy: X is part of Y
    HAS_PROPERTY = "has_property"    # Attribution: X has property Y
    CAUSES = "causes"                # Causation: X causes Y
    LOCATED_IN = "located_in"        # Location: X is in Y
    CREATED_BY = "created_by"        # Creation: X was created by Y
    RELATED_TO = "related_to"        # Generic relation
    # Action relations (from SVO)
    ACTION = "action"                # Subject performs action on Object
    # Temporal relations
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"


@dataclass
class Entity:
    """
    Node in the knowledge graph representing an entity.

    Attributes:
        id: Unique identifier (hash of normalized name)
        name: Original entity name
        normalized_name: Lowercase, cleaned name for matching
        entity_type: Optional type classification
        embedding: Dense vector representation
        metadata: Additional attributes
        mention_count: Number of times entity was mentioned
    """
    id: str
    name: str
    normalized_name: str
    entity_type: Optional[str] = None
    embedding: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    mention_count: int = 1

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Entity):
            return self.id == other.id
        return False


@dataclass
class Relation:
    """
    Edge in the knowledge graph representing a relation.

    Attributes:
        relation_type: Type of relation (from RelationType)
        label: Human-readable label (e.g., verb phrase)
        weight: Confidence/strength of relation
        metadata: Additional attributes (source, timestamp, etc.)
    """
    relation_type: RelationType
    label: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Triple:
    """
    A knowledge triple: (subject, predicate, object).

    Represents a fact extracted from text.
    """
    subject: Entity
    predicate: Relation
    object: Entity
    source_text: Optional[str] = None
    confidence: float = 1.0


@dataclass
class SubgraphResult:
    """
    Result of a subgraph query.

    Contains the relevant entities, relations, and paths
    for use in reasoning context.
    """
    center_entity: Entity
    entities: List[Entity]
    triples: List[Triple]
    paths: List[List[Tuple[Entity, Relation, Entity]]]
    depth: int
    relevance_scores: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# SVO Extractor
# ============================================================================

class SVOExtractor:
    """
    Simple Subject-Verb-Object extractor using rule-based patterns.

    For production use, consider:
    - spaCy dependency parsing
    - Stanford CoreNLP
    - Hugging Face NER models

    This implementation uses regex patterns for lightweight extraction
    without heavy NLP dependencies.
    """

    # Common auxiliary verbs and modals
    AUX_VERBS = {
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'has', 'have', 'had', 'having',
        'do', 'does', 'did',
        'will', 'would', 'shall', 'should',
        'may', 'might', 'must', 'can', 'could',
        'là', 'được', 'bị', 'có', 'đã', 'sẽ', 'đang',  # Vietnamese
    }

    # Common prepositions for relation extraction
    PREPOSITIONS = {
        'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to',
        'of', 'about', 'through', 'during', 'before', 'after',
        'trong', 'trên', 'tại', 'bởi', 'cho', 'với', 'từ', 'đến',  # Vietnamese
    }

    # Relation type mappings from verbs/patterns
    VERB_TO_RELATION = {
        # IS_A relations
        'is': RelationType.IS_A,
        'are': RelationType.IS_A,
        'was': RelationType.IS_A,
        'were': RelationType.IS_A,
        'là': RelationType.IS_A,
        # PART_OF relations
        'part of': RelationType.PART_OF,
        'belongs to': RelationType.PART_OF,
        'thuộc về': RelationType.PART_OF,
        # CAUSES relations
        'causes': RelationType.CAUSES,
        'leads to': RelationType.CAUSES,
        'results in': RelationType.CAUSES,
        'gây ra': RelationType.CAUSES,
        # LOCATED_IN relations
        'located in': RelationType.LOCATED_IN,
        'lives in': RelationType.LOCATED_IN,
        'situated in': RelationType.LOCATED_IN,
        'ở': RelationType.LOCATED_IN,
        'nằm ở': RelationType.LOCATED_IN,
        # CREATED_BY relations
        'created by': RelationType.CREATED_BY,
        'developed by': RelationType.CREATED_BY,
        'invented by': RelationType.CREATED_BY,
        'written by': RelationType.CREATED_BY,
        'được tạo bởi': RelationType.CREATED_BY,
        'do': RelationType.CREATED_BY,  # Vietnamese: "do X phát triển"
    }

    def __init__(self):
        """Initialize the SVO extractor."""
        # Compile patterns for efficiency
        self._sentence_pattern = re.compile(r'[.!?]+')
        self._word_pattern = re.compile(r'\b\w+\b', re.UNICODE)

        # Simple POS-like patterns (noun phrases, verb phrases)
        # Pattern: Capitalized words or multi-word proper nouns
        self._noun_phrase_pattern = re.compile(
            r'\b([A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]'
            r'[a-zàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]*'
            r'(?:\s+[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]'
            r'[a-zàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]*)*)\b',
            re.UNICODE
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = self._sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_noun_phrases(self, sentence: str) -> List[str]:
        """Extract potential noun phrases (entities) from sentence."""
        # Find capitalized phrases (proper nouns)
        proper_nouns = self._noun_phrase_pattern.findall(sentence)

        # Also extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', sentence)
        quoted += re.findall(r"'([^']+)'", sentence)

        # Combine and deduplicate
        phrases = list(set(proper_nouns + quoted))

        # Filter very short or common words
        phrases = [p for p in phrases if len(p) > 1 and p.lower() not in self.AUX_VERBS]

        return phrases

    def _find_verb_phrase(self, sentence: str) -> Optional[Tuple[str, RelationType]]:
        """Find the main verb/relation in the sentence."""
        sentence_lower = sentence.lower()

        # Check for known relation patterns (longest match first)
        for pattern, rel_type in sorted(
            self.VERB_TO_RELATION.items(),
            key=lambda x: -len(x[0])
        ):
            if pattern in sentence_lower:
                return (pattern, rel_type)

        # Fallback: find any verb-like word
        words = self._word_pattern.findall(sentence_lower)
        for word in words:
            if word not in self.AUX_VERBS and word not in self.PREPOSITIONS:
                # Heuristic: verbs often end in -ed, -ing, -s for English
                if (word.endswith('ed') or word.endswith('ing') or
                    word.endswith('es') or word.endswith('ates')):
                    return (word, RelationType.ACTION)

        return None

    def extract(self, text: str) -> List[Triple]:
        """
        Extract SVO triples from text.

        Algorithm:
        1. Split text into sentences
        2. For each sentence:
           a. Extract noun phrases as potential entities
           b. Find verb phrase as relation
           c. Construct triples from SVO patterns

        Args:
            text: Input text to extract from

        Returns:
            List of extracted Triple objects
        """
        text = self._normalize_text(text)
        sentences = self._split_sentences(text)
        triples = []

        for sentence in sentences:
            # Extract entities
            entities = self._extract_noun_phrases(sentence)
            if len(entities) < 2:
                # Need at least subject and object
                continue

            # Find relation
            verb_info = self._find_verb_phrase(sentence)
            if not verb_info:
                verb_label = "related_to"
                rel_type = RelationType.RELATED_TO
            else:
                verb_label, rel_type = verb_info

            # Create triples from entity pairs
            # Heuristic: first entity is subject, others are objects
            # (This is simplified; real NLP would use dependency parsing)
            subject_name = entities[0]
            subject = self._create_entity(subject_name)

            for obj_name in entities[1:]:
                obj = self._create_entity(obj_name)
                relation = Relation(
                    relation_type=rel_type,
                    label=verb_label,
                    weight=0.8,  # Default confidence
                    metadata={"source_sentence": sentence}
                )

                triple = Triple(
                    subject=subject,
                    predicate=relation,
                    object=obj,
                    source_text=sentence,
                    confidence=0.8
                )
                triples.append(triple)

        return triples

    def _create_entity(self, name: str) -> Entity:
        """Create an Entity from a name."""
        normalized = name.lower().strip()
        entity_id = hashlib.md5(normalized.encode()).hexdigest()[:12]

        return Entity(
            id=entity_id,
            name=name,
            normalized_name=normalized
        )


# ============================================================================
# Knowledge Graph RAG
# ============================================================================

class KnowledgeGraphRAG(nn.Module):
    """
    Graph-enhanced Retrieval Augmented Generation (GraphRAG).

    Combines a networkx knowledge graph with embedding-based retrieval
    for context-aware reasoning.

    Key Features:
    - Automatic SVO extraction from documents
    - Subgraph queries with configurable depth
    - Entity embedding for semantic similarity
    - Path-based context for reasoning
    - Transformer prompt integration

    Graph Traversal Logic:
    The query_subgraph method uses BFS (Breadth-First Search) to explore
    the graph from a center entity. At each depth level:
    1. Collect all neighbors of current frontier
    2. Score relevance based on edge weights and embedding similarity
    3. Expand frontier to unvisited high-relevance neighbors
    4. Track paths for explainability

    This provides richer context than simple keyword retrieval by
    capturing relational structure.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        max_entities: int = 100000,
        similarity_threshold: float = 0.7,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the Knowledge Graph.

        Args:
            embedding_dim: Dimension of entity embeddings
            max_entities: Maximum number of entities to store
            similarity_threshold: Threshold for entity matching
            device: Torch device for embeddings
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_entities = max_entities
        self.similarity_threshold = similarity_threshold
        self.device = device or torch.device('cpu')

        # NetworkX directed graph
        self.graph = nx.DiGraph()

        # Entity storage (id -> Entity)
        self.entities: Dict[str, Entity] = {}

        # Name index for fast lookup (normalized_name -> entity_id)
        self.name_index: Dict[str, str] = {}

        # SVO extractor
        self.extractor = SVOExtractor()

        # Entity embedding layer (learnable)
        self.entity_embeddings = nn.Embedding(max_entities, embedding_dim)
        self._entity_id_to_idx: Dict[str, int] = {}
        self._next_idx = 0

        # Embedding encoder for text
        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        # Initialize weights
        nn.init.xavier_uniform_(self.entity_embeddings.weight)

    @property
    def num_entities(self) -> int:
        """Number of entities in the graph."""
        return len(self.entities)

    @property
    def num_relations(self) -> int:
        """Number of relations (edges) in the graph."""
        return self.graph.number_of_edges()

    def _get_or_create_entity_idx(self, entity_id: str) -> int:
        """Get embedding index for entity, creating if needed."""
        if entity_id not in self._entity_id_to_idx:
            if self._next_idx >= self.max_entities:
                raise RuntimeError(f"Maximum entities ({self.max_entities}) exceeded")
            self._entity_id_to_idx[entity_id] = self._next_idx
            self._next_idx += 1
        return self._entity_id_to_idx[entity_id]

    def _find_matching_entity(self, name: str) -> Optional[Entity]:
        """
        Find existing entity by name using exact and fuzzy matching.

        Args:
            name: Entity name to search for

        Returns:
            Matching Entity or None
        """
        normalized = name.lower().strip()

        # Exact match
        if normalized in self.name_index:
            return self.entities[self.name_index[normalized]]

        # TODO: Add fuzzy matching using embeddings if needed
        return None

    def _add_entity(self, entity: Entity) -> Entity:
        """
        Add entity to graph, merging with existing if found.

        Args:
            entity: Entity to add

        Returns:
            The added or merged entity
        """
        # Check for existing entity
        existing = self._find_matching_entity(entity.name)
        if existing:
            # Merge: increment mention count
            existing.mention_count += 1
            return existing

        # Add new entity
        self.entities[entity.id] = entity
        self.name_index[entity.normalized_name] = entity.id

        # Create embedding index
        idx = self._get_or_create_entity_idx(entity.id)

        # Add to graph
        self.graph.add_node(entity.id, entity=entity)

        return entity

    def add_triple(self, triple: Triple) -> None:
        """
        Add a knowledge triple to the graph.

        Args:
            triple: Triple (subject, predicate, object) to add
        """
        # Add entities
        subject = self._add_entity(triple.subject)
        obj = self._add_entity(triple.object)

        # Add edge (relation)
        edge_data = {
            'relation': triple.predicate,
            'weight': triple.predicate.weight,
            'label': triple.predicate.label,
            'type': triple.predicate.relation_type.value,
            'source_text': triple.source_text,
        }

        # If edge exists, update weight (accumulate evidence)
        if self.graph.has_edge(subject.id, obj.id):
            existing = self.graph.edges[subject.id, obj.id]
            existing['weight'] = min(1.0, existing['weight'] + 0.1)
        else:
            self.graph.add_edge(subject.id, obj.id, **edge_data)

    def add_document(self, text: str, metadata: Optional[Dict] = None) -> int:
        """
        Add document to the knowledge graph by extracting triples.

        Algorithm:
        1. Use SVO extractor to get triples from text
        2. Add each triple to the graph
        3. Update entity embeddings based on context

        Args:
            text: Document text
            metadata: Optional metadata for provenance

        Returns:
            Number of triples extracted and added
        """
        # Extract triples
        triples = self.extractor.extract(text)

        # Add metadata to triples
        if metadata:
            for triple in triples:
                triple.predicate.metadata.update(metadata)

        # Add to graph
        for triple in triples:
            self.add_triple(triple)

        logger.debug(f"Added {len(triples)} triples from document")
        return len(triples)

    def query_subgraph(
        self,
        topic: str,
        depth: int = 2,
        max_neighbors: int = 10,
        min_relevance: float = 0.3,
    ) -> SubgraphResult:
        """
        Query subgraph around a topic with neighbor expansion.

        Algorithm (BFS with relevance scoring):
        1. Find center entity matching topic
        2. Initialize frontier with center entity
        3. For each depth level:
           a. Get all neighbors of frontier entities
           b. Score each neighbor by edge weight and path relevance
           c. Select top-k most relevant neighbors
           d. Add to result and update frontier
        4. Collect all paths for explainability

        This captures not just the topic entity but its relational context,
        enabling more informed reasoning.

        Args:
            topic: Topic string to search for
            depth: How many hops from center (default: 2)
            max_neighbors: Max neighbors per entity per level
            min_relevance: Minimum relevance score to include

        Returns:
            SubgraphResult with entities, triples, and paths
        """
        # Find center entity
        center = self._find_matching_entity(topic)
        if center is None:
            # Create placeholder if not found
            logger.warning(f"Topic '{topic}' not found in graph, creating placeholder")
            center = Entity(
                id=hashlib.md5(topic.lower().encode()).hexdigest()[:12],
                name=topic,
                normalized_name=topic.lower()
            )
            return SubgraphResult(
                center_entity=center,
                entities=[center],
                triples=[],
                paths=[],
                depth=depth,
                relevance_scores={center.id: 1.0}
            )

        # BFS traversal
        visited: Set[str] = {center.id}
        frontier: Set[str] = {center.id}
        collected_entities: List[Entity] = [center]
        collected_triples: List[Triple] = []
        collected_paths: List[List[Tuple[Entity, Relation, Entity]]] = []
        relevance_scores: Dict[str, float] = {center.id: 1.0}

        for current_depth in range(depth):
            next_frontier: Set[str] = set()
            depth_decay = 0.7 ** current_depth  # Relevance decays with distance

            for entity_id in frontier:
                # Get neighbors (both outgoing and incoming edges)
                out_neighbors = list(self.graph.successors(entity_id))
                in_neighbors = list(self.graph.predecessors(entity_id))

                all_neighbors = []

                # Process outgoing edges
                for neighbor_id in out_neighbors:
                    if neighbor_id not in visited:
                        edge_data = self.graph.edges[entity_id, neighbor_id]
                        relevance = edge_data.get('weight', 0.5) * depth_decay
                        all_neighbors.append((neighbor_id, relevance, 'out', edge_data))

                # Process incoming edges
                for neighbor_id in in_neighbors:
                    if neighbor_id not in visited:
                        edge_data = self.graph.edges[neighbor_id, entity_id]
                        relevance = edge_data.get('weight', 0.5) * depth_decay
                        all_neighbors.append((neighbor_id, relevance, 'in', edge_data))

                # Sort by relevance and take top-k
                all_neighbors.sort(key=lambda x: -x[1])
                top_neighbors = all_neighbors[:max_neighbors]

                for neighbor_id, relevance, direction, edge_data in top_neighbors:
                    if relevance < min_relevance:
                        continue

                    visited.add(neighbor_id)
                    next_frontier.add(neighbor_id)

                    # Add entity
                    neighbor_entity = self.entities.get(neighbor_id)
                    if neighbor_entity:
                        collected_entities.append(neighbor_entity)
                        relevance_scores[neighbor_id] = relevance

                        # Create triple
                        source_entity = self.entities.get(entity_id)
                        if source_entity:
                            relation = edge_data.get('relation', Relation(
                                relation_type=RelationType.RELATED_TO,
                                label=edge_data.get('label', 'related'),
                                weight=edge_data.get('weight', 0.5)
                            ))

                            if direction == 'out':
                                triple = Triple(
                                    subject=source_entity,
                                    predicate=relation,
                                    object=neighbor_entity,
                                    source_text=edge_data.get('source_text')
                                )
                            else:
                                triple = Triple(
                                    subject=neighbor_entity,
                                    predicate=relation,
                                    object=source_entity,
                                    source_text=edge_data.get('source_text')
                                )

                            collected_triples.append(triple)
                            # Track path
                            collected_paths.append([
                                (triple.subject, triple.predicate, triple.object)
                            ])

            frontier = next_frontier

        return SubgraphResult(
            center_entity=center,
            entities=collected_entities,
            triples=collected_triples,
            paths=collected_paths,
            depth=depth,
            relevance_scores=relevance_scores
        )

    def format_for_prompt(
        self,
        subgraph: SubgraphResult,
        max_triples: int = 20,
        include_paths: bool = True,
    ) -> str:
        """
        Format subgraph as context for Transformer prompt.

        Creates a structured text representation that can be prepended
        to the model's input for knowledge-grounded generation.

        Format:
        ```
        [Knowledge Context]
        Topic: {center_entity}

        Facts:
        - {subject} {relation} {object}
        - ...

        Related Concepts:
        - {entity}: relevance {score}
        ```

        Args:
            subgraph: SubgraphResult from query_subgraph
            max_triples: Maximum triples to include
            include_paths: Whether to include reasoning paths

        Returns:
            Formatted string for prompt injection
        """
        lines = ["[Knowledge Context]"]
        lines.append(f"Topic: {subgraph.center_entity.name}")
        lines.append("")

        # Add facts (triples)
        if subgraph.triples:
            lines.append("Facts:")
            # Sort by relevance
            sorted_triples = sorted(
                subgraph.triples,
                key=lambda t: subgraph.relevance_scores.get(t.object.id, 0),
                reverse=True
            )[:max_triples]

            for triple in sorted_triples:
                fact = f"- {triple.subject.name} {triple.predicate.label} {triple.object.name}"
                lines.append(fact)
            lines.append("")

        # Add related concepts with relevance
        if len(subgraph.entities) > 1:
            lines.append("Related Concepts:")
            # Sort by relevance, exclude center
            sorted_entities = sorted(
                [(e, subgraph.relevance_scores.get(e.id, 0))
                 for e in subgraph.entities if e.id != subgraph.center_entity.id],
                key=lambda x: -x[1]
            )[:10]

            for entity, score in sorted_entities:
                lines.append(f"- {entity.name} (relevance: {score:.2f})")
            lines.append("")

        # Add reasoning paths if requested
        if include_paths and subgraph.paths:
            lines.append("Reasoning Paths:")
            for i, path in enumerate(subgraph.paths[:5]):
                path_str = " -> ".join([
                    f"{s.name} --[{r.label}]--> {o.name}"
                    for s, r, o in path
                ])
                lines.append(f"  {i+1}. {path_str}")

        lines.append("[/Knowledge Context]")
        return "\n".join(lines)

    def get_entity_embedding(self, entity_id: str) -> Optional[torch.Tensor]:
        """Get embedding for an entity."""
        if entity_id not in self._entity_id_to_idx:
            return None
        idx = self._entity_id_to_idx[entity_id]
        return self.entity_embeddings.weight[idx]

    def semantic_search(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 10,
    ) -> List[Tuple[Entity, float]]:
        """
        Search for entities by embedding similarity.

        Args:
            query_embedding: Query vector [embedding_dim]
            top_k: Number of results

        Returns:
            List of (Entity, similarity_score) tuples
        """
        if self._next_idx == 0:
            return []

        # Get all entity embeddings
        entity_embs = self.entity_embeddings.weight[:self._next_idx]

        # Normalize
        query_norm = F.normalize(query_embedding.unsqueeze(0), dim=-1)
        entity_norm = F.normalize(entity_embs, dim=-1)

        # Cosine similarity
        similarities = torch.mm(query_norm, entity_norm.T).squeeze(0)

        # Top-k
        top_scores, top_indices = torch.topk(similarities, min(top_k, self._next_idx))

        # Map back to entities
        idx_to_id = {v: k for k, v in self._entity_id_to_idx.items()}
        results = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            entity_id = idx_to_id.get(idx)
            if entity_id and entity_id in self.entities:
                results.append((self.entities[entity_id], score))

        return results

    def save(self, path: str) -> None:
        """Save graph to disk."""
        import json
        from pathlib import Path

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save graph structure
        nx.write_gpickle(self.graph, save_path / "graph.gpickle")

        # Save entities
        entities_data = {
            eid: {
                'id': e.id,
                'name': e.name,
                'normalized_name': e.normalized_name,
                'entity_type': e.entity_type,
                'metadata': e.metadata,
                'mention_count': e.mention_count
            }
            for eid, e in self.entities.items()
        }
        with open(save_path / "entities.json", 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, ensure_ascii=False, indent=2)

        # Save embeddings
        torch.save({
            'entity_embeddings': self.entity_embeddings.state_dict(),
            'entity_id_to_idx': self._entity_id_to_idx,
            'next_idx': self._next_idx,
        }, save_path / "embeddings.pt")

        logger.info(f"Saved graph to {save_path}")

    def load(self, path: str) -> None:
        """Load graph from disk."""
        import json
        from pathlib import Path

        load_path = Path(path)

        # Load graph structure
        self.graph = nx.read_gpickle(load_path / "graph.gpickle")

        # Load entities
        with open(load_path / "entities.json", 'r', encoding='utf-8') as f:
            entities_data = json.load(f)

        self.entities = {}
        self.name_index = {}
        for eid, data in entities_data.items():
            entity = Entity(
                id=data['id'],
                name=data['name'],
                normalized_name=data['normalized_name'],
                entity_type=data.get('entity_type'),
                metadata=data.get('metadata', {}),
                mention_count=data.get('mention_count', 1)
            )
            self.entities[eid] = entity
            self.name_index[entity.normalized_name] = eid

        # Load embeddings
        emb_data = torch.load(load_path / "embeddings.pt")
        self.entity_embeddings.load_state_dict(emb_data['entity_embeddings'])
        self._entity_id_to_idx = emb_data['entity_id_to_idx']
        self._next_idx = emb_data['next_idx']

        logger.info(f"Loaded graph from {load_path}: {self.num_entities} entities, {self.num_relations} relations")

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(1, self.num_entities),
            'num_connected_components': nx.number_weakly_connected_components(self.graph),
            'density': nx.density(self.graph),
        }


# ============================================================================
# Integration Helper
# ============================================================================

def create_graphrag_context(
    graph: KnowledgeGraphRAG,
    query: str,
    depth: int = 2,
    max_context_tokens: int = 500,
) -> str:
    """
    Helper function to create GraphRAG context for a query.

    Usage in Transformer pipeline:
    ```python
    graph = KnowledgeGraphRAG()
    graph.add_document("Einstein developed relativity...")

    # In generation
    query = "What did Einstein contribute to physics?"
    context = create_graphrag_context(graph, query)
    full_prompt = context + "\n\n" + query
    output = model.generate(full_prompt)
    ```

    Args:
        graph: KnowledgeGraphRAG instance
        query: User query
        depth: Subgraph depth
        max_context_tokens: Approximate max tokens for context

    Returns:
        Formatted context string
    """
    # Extract key entities from query
    extractor = SVOExtractor()
    entities = extractor._extract_noun_phrases(query)

    if not entities:
        # Fallback: use first significant words
        words = query.split()
        entities = [w for w in words if len(w) > 3 and w[0].isupper()][:3]

    if not entities:
        return ""

    # Query subgraph for each entity and merge
    all_contexts = []
    for entity in entities[:3]:  # Limit to top 3 entities
        subgraph = graph.query_subgraph(entity, depth=depth)
        if subgraph.triples:
            context = graph.format_for_prompt(subgraph, max_triples=10)
            all_contexts.append(context)

    return "\n\n".join(all_contexts)
