"""Reflection manager for abstracting recent interactions into insights."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Iterable, List, Optional, Protocol, Sequence
import json
import re

from .generative_memory import MemoryObject, MemoryStream


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class MemorySink(Protocol):
    def add_memory(
        self,
        content: str,
        *,
        importance_score: Optional[float] = None,
        related_nodes: Optional[Iterable[str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> MemoryObject:
        ...


@dataclass
class Interaction:
    timestamp: datetime
    user_text: str
    assistant_text: str


@dataclass
class ReflexionConfig:
    buffer_size: int = 20
    raw_importance: float = 0.4
    insight_importance: float = 0.85
    max_insights: int = 3
    prompt_template: str = (
        "Here are the last {n} interactions. Identify {k} high-level patterns "
        "or facts about the user (Zynther) or the world. Do not just summarize.\n\n"
        "Interactions:\n{interactions}\n\n"
        "Insights:"
    )


class ReflexionManager:
    """Buffer interactions and synthesize insights asynchronously."""

    def __init__(
        self,
        memory_store: Optional[MemorySink] = None,
        *,
        config: Optional[ReflexionConfig] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        self.memory_store = memory_store or MemoryStream()
        self.config = config or ReflexionConfig()
        self._buffer: List[Interaction] = []
        self._lock = Lock()
        self._executor = executor or ThreadPoolExecutor(max_workers=2)
        self._pending: List[Future[List[MemoryObject]]] = []

    def add_interaction(
        self,
        user_text: str,
        assistant_text: str,
        *,
        llm_fn: Optional[Callable[[str], str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[Future[List[MemoryObject]]]:
        interaction = Interaction(
            timestamp=timestamp or _utc_now(),
            user_text=user_text,
            assistant_text=assistant_text,
        )
        with self._lock:
            self._buffer.append(interaction)
            self.memory_store.add_memory(
                content=f"User: {user_text}\nAssistant: {assistant_text}",
                importance_score=self.config.raw_importance,
                timestamp=interaction.timestamp,
            )
            should_reflect = len(self._buffer) >= self.config.buffer_size
            if not should_reflect:
                return None
            snapshot = list(self._buffer)
            self._buffer.clear()

        if llm_fn is None:
            self._buffer = snapshot + self._buffer
            return None
        future = self._executor.submit(self._reflect_from_snapshot, snapshot, llm_fn)
        self._pending.append(future)
        return future

    def reflect(self, llm_fn: Callable[[str], str]) -> List[MemoryObject]:
        snapshot = self._drain_buffer()
        if not snapshot:
            return []
        return self._reflect_from_snapshot(snapshot, llm_fn)

    def reflect_async(self, llm_fn: Callable[[str], str]) -> Optional[Future[List[MemoryObject]]]:
        snapshot = self._drain_buffer()
        if not snapshot:
            return None
        future = self._executor.submit(self._reflect_from_snapshot, snapshot, llm_fn)
        self._pending.append(future)
        return future

    def pending(self) -> List[Future[List[MemoryObject]]]:
        return list(self._pending)

    def _drain_buffer(self) -> List[Interaction]:
        with self._lock:
            if not self._buffer:
                return []
            snapshot = list(self._buffer)
            self._buffer.clear()
            return snapshot

    def _reflect_from_snapshot(
        self,
        snapshot: Sequence[Interaction],
        llm_fn: Callable[[str], str],
    ) -> List[MemoryObject]:
        prompt = self._build_prompt(snapshot)
        response = llm_fn(prompt)
        insights = self._parse_insights(response)
        stored: List[MemoryObject] = []
        for insight in insights[: self.config.max_insights]:
            stored.append(
                self.memory_store.add_memory(
                    content=insight,
                    importance_score=self.config.insight_importance,
                )
            )
        return stored

    def _build_prompt(self, snapshot: Sequence[Interaction]) -> str:
        interactions = []
        for interaction in snapshot[-self.config.buffer_size :]:
            ts = interaction.timestamp.isoformat()
            interactions.append(
                f"- [{ts}] User: {interaction.user_text} | Assistant: {interaction.assistant_text}"
            )
        interactions_block = "\n".join(interactions)
        return self.config.prompt_template.format(
            n=min(len(snapshot), self.config.buffer_size),
            k=self.config.max_insights,
            interactions=interactions_block,
        )

    @staticmethod
    def _parse_insights(response: str) -> List[str]:
        if not response:
            return []
        array_match = re.search(r"\[[^\[\]]+\]", response, flags=re.DOTALL)
        if array_match:
            try:
                payload = json.loads(array_match.group(0))
                if isinstance(payload, list):
                    return [str(item).strip() for item in payload if str(item).strip()]
            except json.JSONDecodeError:
                pass

        insights = []
        for line in response.splitlines():
            text = line.strip()
            if not text:
                continue
            text = re.sub(r"^[-*]\s+", "", text)
            text = re.sub(r"^\d+[.)]\s+", "", text)
            if text:
                insights.append(text)

        if not insights:
            stripped = response.strip()
            if stripped:
                insights.append(stripped)
        return insights
