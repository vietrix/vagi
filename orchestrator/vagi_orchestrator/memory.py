from __future__ import annotations

import os
import re
from dataclasses import dataclass
from threading import Lock
from pathlib import Path
from typing import Any

import httpx

_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384
_MODEL_LOCK = Lock()
_MODEL_INSTANCE: Any | None = None
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
_DEFINITION_QUERY_RE = re.compile(r"(?:\bl[àa]\s*g[ìi]\b)|(?:\bwhat\s+is\b)", flags=re.IGNORECASE)
_COMMAND_LIKE_RE = re.compile(
    r"(?:\b(?:cargo|pip|python|uvicorn|npm|pnpm|yarn|go|rustc)\b)|(?:http://|https://)|(?:--\w+)|(?:[/\\])",
    flags=re.IGNORECASE,
)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


@dataclass(slots=True, frozen=True)
class MemoryHit:
    text: str
    score: float


@dataclass(slots=True, frozen=True)
class MemoryAnswer:
    answer: str
    hits: list[MemoryHit]


def _split_paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n+", normalized)]
    return [chunk for chunk in chunks if chunk]


class EmbeddingService:
    @classmethod
    def _get_model(cls) -> Any:
        global _MODEL_INSTANCE
        if _MODEL_INSTANCE is None:
            with _MODEL_LOCK:
                if _MODEL_INSTANCE is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ModuleNotFoundError as exc:
                        if exc.name == "sentence_transformers":
                            raise RuntimeError(
                                "Missing dependency `sentence-transformers`. "
                                "Run: pip install -r orchestrator/requirements.txt"
                            ) from exc
                        raise

                    _MODEL_INSTANCE = SentenceTransformer(_MODEL_NAME)
        return _MODEL_INSTANCE

    @classmethod
    def embed(cls, text: str) -> list[float]:
        normalized = text.strip()
        if not normalized:
            raise ValueError("text must not be empty")

        model = cls._get_model()
        vector = model.encode(normalized, convert_to_numpy=True)
        values = [float(value) for value in vector.tolist()]
        if len(values) != _EMBEDDING_DIM:
            raise ValueError(
                f"embedding dimension mismatch: expected {_EMBEDDING_DIM}, got {len(values)}"
            )
        return values


class MemoryClient:
    def __init__(self, kernel_url: str | None = None, timeout: float = 120.0) -> None:
        self._kernel_url = self._normalize_kernel_url(kernel_url)
        self._timeout = timeout

    @staticmethod
    def _normalize_kernel_url(url: str | None) -> str:
        if url:
            return url.rstrip("/")
        return "http://127.0.0.1:17070"

    def add_document(self, text: str) -> bool:
        normalized = text.strip()
        if not normalized:
            raise ValueError("text must not be empty")

        vector = EmbeddingService.embed(normalized)
        response = httpx.post(
            f"{self._kernel_url}/internal/memory/add",
            json={"text": normalized, "vector": vector},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return bool(payload.get("id"))

    def ingest_file(self, file_path: str | Path) -> dict[str, int]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"path is not a file: {path}")

        text = path.read_text(encoding="utf-8")
        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            raise ValueError("file does not contain non-empty paragraphs")

        total = len(paragraphs)
        success = 0
        failed = 0
        for paragraph in paragraphs:
            try:
                if self.add_document(paragraph):
                    success += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
        return {
            "total": total,
            "success": success,
            "failed": failed,
        }

    def retrieve_hits(self, query: str, top_k: int = 3) -> list[MemoryHit]:
        normalized = query.strip()
        if not normalized:
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        vector = EmbeddingService.embed(normalized)
        request_top_k = max(top_k * 8, top_k)
        response = httpx.post(
            f"{self._kernel_url}/internal/memory/search",
            json={"vector": vector, "top_k": request_top_k},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        dedup_hits: dict[str, MemoryHit] = {}
        for item in results:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            score_raw = item.get("score")
            if not isinstance(text, str):
                continue
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                continue
            normalized_text = re.sub(r"\s+", " ", text).strip()
            if not normalized_text:
                continue
            previous = dedup_hits.get(normalized_text)
            current = MemoryHit(text=text, score=score)
            if previous is None or current.score > previous.score:
                dedup_hits[normalized_text] = current
        hits = list(dedup_hits.values())
        hits.sort(key=lambda hit: (hit.score + _quality_adjustment(hit.text), hit.score), reverse=True)
        return hits[:top_k]

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        return [hit.text for hit in self.retrieve_hits(query=query, top_k=top_k)]

    def answer(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.05,
        max_sentences: int = 3,
    ) -> MemoryAnswer:
        if max_sentences <= 0:
            raise ValueError("max_sentences must be greater than 0")
        candidate_top_k = max(top_k * 4, top_k)
        hits = self.retrieve_hits(query=query, top_k=candidate_top_k)
        filtered_hits = [hit for hit in hits if hit.score >= min_score]
        if not filtered_hits:
            return MemoryAnswer(answer="", hits=[])
        answer_text = _compose_answer(query=query, hits=filtered_hits, max_sentences=max_sentences)
        return MemoryAnswer(answer=answer_text, hits=filtered_hits[:top_k])

    def search(self, query: str, top_k: int = 3) -> list[str]:
        return self.retrieve(query=query, top_k=top_k)


def _normalize_text_for_answer(text: str) -> str:
    cleaned_lines: list[str] = []
    in_code_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        cleaned_lines.append(stripped)
    normalized = " ".join(cleaned_lines)
    normalized = normalized.replace("**", "").replace("`", "")
    normalized = re.sub(r"\b\d+\.\s*", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(text)}


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _compose_answer(query: str, hits: list[MemoryHit], max_sentences: int) -> str:
    query_tokens = _tokenize(query)
    is_definition_query = bool(_DEFINITION_QUERY_RE.search(query))
    scored_sentences: list[tuple[float, str]] = []

    for hit in hits:
        normalized_text = _normalize_text_for_answer(hit.text)
        if not normalized_text:
            continue
        sentences = _split_sentences(normalized_text)
        if not sentences:
            sentences = [normalized_text]

        for sentence in sentences:
            sentence_tokens = _tokenize(sentence)
            if is_definition_query and _COMMAND_LIKE_RE.search(sentence):
                continue
            overlap = (
                len(query_tokens & sentence_tokens) / max(1, len(query_tokens))
                if query_tokens
                else 0.0
            )
            length_bonus = min(len(sentence) / 220.0, 1.0)
            score = (hit.score * 0.65) + (overlap * 0.3) + (length_bonus * 0.05)
            if len(sentence) >= 24:
                scored_sentences.append((score, sentence))

    if not scored_sentences:
        fallback = _normalize_text_for_answer(hits[0].text)
        return fallback[:400]

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for _, sentence in scored_sentences:
        normalized_sentence = sentence.lower()
        if normalized_sentence in seen:
            continue
        seen.add(normalized_sentence)
        selected.append(sentence)
        if len(selected) >= max_sentences:
            break

    return " ".join(selected)


def _quality_adjustment(text: str) -> float:
    raw = text.strip()
    normalized = _normalize_text_for_answer(text)
    if not normalized:
        return -0.3

    adjustment = 0.0
    lowered = normalized.lower()
    if raw.startswith("#"):
        adjustment -= 0.2
    if _COMMAND_LIKE_RE.search(normalized):
        adjustment -= 0.18
    if len(normalized) < 40:
        adjustment -= 0.12
    if len(normalized) >= 70 and not _COMMAND_LIKE_RE.search(normalized):
        adjustment += 0.06
    if any(token in lowered for token in ("là", "la ", "gồm", "bao gồm", "kiến trúc", "hệ thống")):
        adjustment += 0.04
    return adjustment
