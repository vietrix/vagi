from __future__ import annotations

import re
from threading import Lock
from pathlib import Path
from typing import Any

import httpx

_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384
_MODEL_LOCK = Lock()
_MODEL_INSTANCE: Any | None = None


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
                    from sentence_transformers import SentenceTransformer

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
        return "http://127.0.0.1:7070"

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

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        normalized = query.strip()
        if not normalized:
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        vector = EmbeddingService.embed(normalized)
        response = httpx.post(
            f"{self._kernel_url}/internal/memory/search",
            json={"vector": vector, "top_k": top_k},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        return [
            item["text"]
            for item in results
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]

    def search(self, query: str, top_k: int = 3) -> list[str]:
        return self.retrieve(query=query, top_k=top_k)
