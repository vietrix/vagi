from __future__ import annotations

import asyncio
from pathlib import Path

from vagi_orchestrator.reasoning import Reasoner
from vagi_orchestrator.store import EpisodeStore


class FakeKernel:
    def __init__(self) -> None:
        self._attempt = 0

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
    metrics = store.metrics()
    assert metrics["total_episodes"] == 1
    store.close()
