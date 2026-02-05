"""Generative memory stream with reflection loop inspired by Generative Agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, List, Optional, Sequence, Tuple
import math
import re


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


@dataclass
class MemoryObject:
    """Atomic memory entry."""

    timestamp: datetime
    content: str
    importance_score: float
    related_nodes: List[str] = field(default_factory=list)

    def clamp_importance(self) -> float:
        return max(0.0, min(1.0, float(self.importance_score)))


@dataclass
class RetrievalFunction:
    """Weighted retrieval scoring function."""

    weight_recency: float = 1.0
    weight_importance: float = 1.0
    weight_relevance: float = 1.0
    recency_decay: float = 0.0001
    relevance_fn: Optional[Callable[[str, str], float]] = None

    def score(
        self,
        memory: MemoryObject,
        query: str,
        now: Optional[datetime] = None,
    ) -> float:
        recency = self._recency_score(memory.timestamp, now=now)
        importance = memory.clamp_importance()
        relevance = self._relevance_score(query, memory.content)
        return (
            self.weight_recency * recency
            + self.weight_importance * importance
            + self.weight_relevance * relevance
        )

    def _recency_score(self, timestamp: datetime, *, now: Optional[datetime]) -> float:
        now = now or _utc_now()
        age_seconds = max(0.0, (now - timestamp).total_seconds())
        return math.exp(-self.recency_decay * age_seconds)

    def _relevance_score(self, query: str, content: str) -> float:
        if self.relevance_fn is not None:
            return float(self.relevance_fn(query, content))
        query_tokens = set(_tokenize(query))
        content_tokens = set(_tokenize(content))
        if not query_tokens or not content_tokens:
            return 0.0
        overlap = len(query_tokens & content_tokens)
        union = len(query_tokens | content_tokens)
        return overlap / max(1, union)


class MemoryStream:
    """In-memory stream of episodic memories with retrieval."""

    def __init__(
        self,
        *,
        retrieval_fn: Optional[RetrievalFunction] = None,
        importance_fn: Optional[Callable[[str], float]] = None,
    ) -> None:
        self._memories: List[MemoryObject] = []
        self._retrieval_fn = retrieval_fn or RetrievalFunction()
        self._importance_fn = importance_fn

    @property
    def memories(self) -> List[MemoryObject]:
        return list(self._memories)

    def add_memory(
        self,
        content: str,
        *,
        importance_score: Optional[float] = None,
        related_nodes: Optional[Iterable[str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> MemoryObject:
        if importance_score is None:
            if self._importance_fn is not None:
                importance_score = float(self._importance_fn(content))
            else:
                importance_score = 0.5
        memory = MemoryObject(
            timestamp=timestamp or _utc_now(),
            content=content,
            importance_score=float(importance_score),
            related_nodes=list(related_nodes) if related_nodes is not None else [],
        )
        self._memories.append(memory)
        return memory

    def recent(self, n: int = 10) -> List[MemoryObject]:
        if n <= 0:
            return []
        return list(self._memories[-n:])

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        now: Optional[datetime] = None,
        return_scores: bool = False,
    ) -> List[MemoryObject] | List[Tuple[MemoryObject, float]]:
        if not self._memories or top_k <= 0:
            return []
        scored = [
            (memory, self._retrieval_fn.score(memory, query, now=now))
            for memory in self._memories
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        scored = scored[:top_k]
        if return_scores:
            return scored
        return [item[0] for item in scored]

    @staticmethod
    def format_memories(memories: Sequence[MemoryObject]) -> str:
        lines = []
        for memory in memories:
            ts = memory.timestamp.isoformat()
            lines.append(f"- [{ts}] {memory.content}")
        return "\n".join(lines)


@dataclass
class ReflectionLoopConfig:
    """Configuration for periodic reflection."""

    reflection_interval_steps: int = 5
    recent_window: int = 20
    min_memories: int = 5
    insight_importance: float = 0.8
    prompt_template: str = (
        "You are a reflective memory module.\n"
        "Given the following recent memories, identify high-level patterns, "
        "goals, and insights. Return 1-3 concise insights.\n\n"
        "Memories:\n{memories}\n\nInsights:"
    )


class ReflectionLoop:
    """Runs periodic reflection over recent memories."""

    def __init__(
        self,
        memory_stream: MemoryStream,
        llm_fn: Optional[Callable[[str], str]],
        *,
        config: Optional[ReflectionLoopConfig] = None,
    ) -> None:
        self.memory_stream = memory_stream
        self.llm_fn = llm_fn
        self.config = config or ReflectionLoopConfig()
        self._last_reflection_step: Optional[int] = None

    def should_reflect(self, step: int) -> bool:
        if step < 0:
            return False
        if self._last_reflection_step is None:
            return step >= self.config.reflection_interval_steps
        return (step - self._last_reflection_step) >= self.config.reflection_interval_steps

    def run_reflection(self, step: int) -> Optional[MemoryObject]:
        if self.llm_fn is None:
            return None
        recent_memories = self.memory_stream.recent(self.config.recent_window)
        if len(recent_memories) < self.config.min_memories:
            return None
        prompt = self._build_prompt(recent_memories)
        insight_text = self.llm_fn(prompt)
        if not insight_text:
            return None
        related_nodes = self._collect_related_nodes(recent_memories)
        importance = self._derive_insight_importance(recent_memories)
        insight = self.memory_stream.add_memory(
            content=insight_text.strip(),
            importance_score=importance,
            related_nodes=related_nodes,
        )
        self._last_reflection_step = step
        return insight

    def maybe_reflect(self, step: int) -> Optional[MemoryObject]:
        if not self.should_reflect(step):
            return None
        return self.run_reflection(step)

    def _build_prompt(self, memories: Sequence[MemoryObject]) -> str:
        memories_block = self.memory_stream.format_memories(memories)
        return self.config.prompt_template.format(memories=memories_block)

    @staticmethod
    def _collect_related_nodes(memories: Sequence[MemoryObject]) -> List[str]:
        nodes: List[str] = []
        seen = set()
        for memory in memories:
            for node in memory.related_nodes:
                if node not in seen:
                    seen.add(node)
                    nodes.append(node)
        return nodes

    def _derive_insight_importance(self, memories: Sequence[MemoryObject]) -> float:
        if not memories:
            return self.config.insight_importance
        avg_importance = sum(m.clamp_importance() for m in memories) / len(memories)
        return min(1.0, max(self.config.insight_importance, avg_importance))
