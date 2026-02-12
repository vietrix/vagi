from __future__ import annotations

import httpx
import pytest

from vagi_orchestrator import memory as memory_module
from vagi_orchestrator.memory import EmbeddingService, MemoryClient


class FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, convert_to_numpy: bool = True) -> FakeVector:
        self.calls.append((text, convert_to_numpy))
        return FakeVector([0.25, 0.5, 0.75])


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.request = httpx.Request("POST", "http://kernel/mock")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "request failed",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request),
            )

    def json(self) -> dict:
        return self._payload


def test_embedding_service_embed_normalizes_and_converts(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel()
    monkeypatch.setattr(
        EmbeddingService,
        "_get_model",
        classmethod(lambda cls: model),
    )

    vector = EmbeddingService.embed("  hello world  ")
    assert vector == [0.25, 0.5, 0.75]
    assert model.calls == [("hello world", True)]


def test_memory_client_add_document_posts_expected_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_capture: dict[str, object] = {}

    def fake_post(url: str, json: dict, timeout: float) -> FakeResponse:
        payload_capture["url"] = url
        payload_capture["json"] = json
        payload_capture["timeout"] = timeout
        return FakeResponse({"id": "f30366bc-5bc8-4a11-9ef8-e4db7d74f57e"})

    monkeypatch.setattr(
        EmbeddingService,
        "embed",
        classmethod(lambda cls, text: [0.1, 0.2, 0.3]),
    )
    monkeypatch.setattr(memory_module.httpx, "post", fake_post)

    client = MemoryClient(kernel_url="http://127.0.0.1:7070", timeout=9.0)
    assert client.add_document("paragraph text") is True
    assert payload_capture["url"] == "http://127.0.0.1:7070/internal/memory/add"
    assert payload_capture["json"] == {
        "text": "paragraph text",
        "vector": [0.1, 0.2, 0.3],
    }
    assert payload_capture["timeout"] == 9.0


def test_memory_client_search_returns_text_list(monkeypatch: pytest.MonkeyPatch) -> None:
    payload_capture: dict[str, object] = {}

    def fake_post(url: str, json: dict, timeout: float) -> FakeResponse:
        payload_capture["url"] = url
        payload_capture["json"] = json
        payload_capture["timeout"] = timeout
        return FakeResponse(
            {
                "results": [
                    {"text": "doc 1", "score": 0.98},
                    {"text": "doc 2", "score": 0.76},
                ]
            }
        )

    monkeypatch.setattr(
        EmbeddingService,
        "embed",
        classmethod(lambda cls, text: [0.9, 0.1]),
    )
    monkeypatch.setattr(memory_module.httpx, "post", fake_post)

    client = MemoryClient(kernel_url="http://kernel")
    hits = client.search("what is login", top_k=2)
    assert hits == ["doc 1", "doc 2"]
    assert payload_capture["url"] == "http://kernel/internal/memory/search"
    assert payload_capture["json"] == {"vector": [0.9, 0.1], "top_k": 2}


def test_memory_client_search_validates_top_k() -> None:
    client = MemoryClient(kernel_url="http://kernel")
    with pytest.raises(ValueError):
        client.search("q", top_k=0)
