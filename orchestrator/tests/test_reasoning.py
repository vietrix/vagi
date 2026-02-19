from __future__ import annotations

import asyncio
from pathlib import Path

from vagi_orchestrator.reasoning import Reasoner
from vagi_orchestrator.memory import MemoryHit
from vagi_orchestrator.store import EpisodeStore


class FakeKernel:
    def __init__(self) -> None:
        self._attempt = 0
        self.last_infer_prompt: str | None = None

    async def init_state(self, session_id: str | None = None) -> dict:
        return {"session_id": session_id or "sid"}

    async def update_state(self, session_id: str, input_text: str) -> dict:
        return {"session_id": session_id, "step": 1}

    async def simulate_world(self, action: str, session_id: str | None = None) -> dict:
        self._attempt += 1
        if self._attempt == 1:
            return {"risk_score": 0.9, "confidence": 0.1, "predicted_effects": [], "causal_path": []}
        return {"risk_score": 0.2, "confidence": 0.8, "predicted_effects": [], "causal_path": []}

    async def verify(
        self,
        patch_ir: str,
        max_loop_iters: int = 2048,
        side_effect_budget: int = 3,
        timeout_ms: int = 80,
    ) -> dict:
        if self._attempt == 1:
            return {"pass": False, "violations": ["infinite_loop_risk"], "wasi_ok": True}
        return {"pass": True, "violations": [], "wasi_ok": True}

    async def model_status(self) -> dict:
        return {"loaded": True, "model_id": "genesis-v0"}

    async def model_infer(self, prompt: str, max_new_tokens: int = 96) -> dict:
        self.last_infer_prompt = prompt
        return {
            "model_id": "genesis-v0",
            "text": "Assistant: Toi se phan tich va de xuat patch an toan.",
            "latency_ms": 11,
        }


class FakeKernelGarbage(FakeKernel):
    async def model_infer(self, prompt: str, max_new_tokens: int = 96) -> dict:
        self.last_infer_prompt = prompt
        return {"model_id": "genesis-v0", "text": "aaaaaaaaaaaaaaaa", "latency_ms": 7}


class FakeKernelMicro(FakeKernel):
    async def micro_ooda_run(
        self,
        input_text: str,
        session_id: str | None = None,
        risk_threshold: float = 0.45,
        max_decide_iters: int = 2,
    ) -> dict:
        return {
            "handled": True,
            "reason": "micro_ooda_success",
            "draft": "echo micro reflex draft",
            "risk_score": 0.11,
            "confidence": 0.93,
            "verifier_pass": True,
            "violations": [],
            "iterations": 1,
        }

    async def homeostasis_status(self) -> dict:
        return {"cortisol": 0.15, "energy": 0.86}


class FakeMemoryClient:
    def __init__(self, hits: list[str]) -> None:
        self._hits = hits
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    def retrieve_hits(self, query: str, top_k: int = 3) -> list[MemoryHit]:
        self.last_query = query
        self.last_top_k = top_k
        return [MemoryHit(text=hit, score=0.9) for hit in self._hits[:top_k]]


def test_reasoner_backtracks_until_safe(tmp_path: Path) -> None:
    store = EpisodeStore(
        db_path=tmp_path / "episodes.db",
        long_term_path=tmp_path / "memory.jsonl",
    )
    reasoner = Reasoner(
        kernel=FakeKernel(),  # type: ignore[arg-type]
        store=store,
        max_decide_iters=4,
        risk_threshold=0.65,
    )
    result = asyncio.run(
        reasoner.run_chat(
            session_id="sid-test",
            messages=[{"role": "user", "content": "please optimize this login flow"}],
        )
    )
    assert "Verifier pass: True" in result["content"]
    assert 0 <= result["trust_score"] <= 1
    trace = result["metadata"]["ooda_trace"]
    assert trace["observe_ok"] is True
    assert trace["orient_ok"] is True
    assert trace["decide_ok"] is True
    assert trace["act_ok"] is True
    assert result["metadata"]["model_runtime"]["used"] is True
    metrics = store.metrics()
    assert metrics["total_episodes"] == 1
    store.close()


def test_reasoner_returns_deterministic_fallback_when_model_garbage(tmp_path: Path) -> None:
    store = EpisodeStore(
        db_path=tmp_path / "episodes.db",
        long_term_path=tmp_path / "memory.jsonl",
    )
    reasoner = Reasoner(
        kernel=FakeKernelGarbage(),  # type: ignore[arg-type]
        store=store,
        max_decide_iters=3,
        risk_threshold=0.7,
    )
    result = asyncio.run(
        reasoner.run_chat(
            session_id="sid-garbage",
            messages=[{"role": "user", "content": "hello"}],
        )
    )
    assert result["content"] == "Insufficient data in kernel state."
    assert result["metadata"]["model_runtime"]["used"] is False
    assert result["metadata"]["model_runtime"]["fallback_reason"] == "garbage_detected"
    store.close()


def test_reasoner_injects_retrieved_context_into_infer_prompt(tmp_path: Path) -> None:
    store = EpisodeStore(
        db_path=tmp_path / "episodes.db",
        long_term_path=tmp_path / "memory.jsonl",
    )
    kernel = FakeKernel()
    memory = FakeMemoryClient(
        hits=[
            "JWT login flow requires nonce validation.",
            "Use Argon2id for password hashing.",
        ]
    )
    reasoner = Reasoner(
        kernel=kernel,  # type: ignore[arg-type]
        store=store,
        memory_client=memory,  # type: ignore[arg-type]
        memory_top_k=2,
        max_decide_iters=4,
        risk_threshold=0.65,
    )
    result = asyncio.run(
        reasoner.run_chat(
            session_id="sid-rag",
            messages=[{"role": "user", "content": "design secure login api"}],
        )
    )

    assert memory.last_query == "design secure login api"
    assert memory.last_top_k == 2
    assert kernel.last_infer_prompt is not None
    assert "Retrieved memory context:" in kernel.last_infer_prompt
    assert "JWT login flow requires nonce validation." in kernel.last_infer_prompt
    retrieval_meta = result["metadata"]["observe"]["retrieval"]
    assert retrieval_meta["used"] is True
    assert retrieval_meta["hits_count"] == 2
    store.close()


def test_reasoner_prefers_micro_reflex_when_available(tmp_path: Path) -> None:
    store = EpisodeStore(
        db_path=tmp_path / "episodes.db",
        long_term_path=tmp_path / "memory.jsonl",
    )
    reasoner = Reasoner(
        kernel=FakeKernelMicro(),  # type: ignore[arg-type]
        store=store,
        max_decide_iters=4,
        risk_threshold=0.65,
    )
    result = asyncio.run(
        reasoner.run_chat(
            session_id="sid-micro",
            messages=[{"role": "user", "content": "fix typo in response"}],
        )
    )
    assert result["metadata"]["ooda_trace"]["mode"] == "micro_reflex"
    assert result["metadata"]["verifier"]["pass"] is True
    assert "Kernel micro-reflex artifact" in result["content"]
    store.close()
