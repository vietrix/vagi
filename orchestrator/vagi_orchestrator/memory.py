from __future__ import annotations

from threading import Lock
from typing import Any

import httpx

_MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL_LOCK = Lock()
_MODEL_INSTANCE: Any | None = None


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
        return [float(value) for value in vector.tolist()]


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

    def search(self, query: str, top_k: int = 3) -> list[str]:
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
