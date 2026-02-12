from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from vagi_orchestrator.app import create_app
from vagi_orchestrator.config import Settings
from vagi_orchestrator.store import EpisodeStore


class FakeKernelPass:
    async def close(self) -> None:
        return None

    async def healthz(self) -> bool:
        return True

    async def init_state(self, session_id: str | None = None) -> dict:
        return {"session_id": session_id or "sid"}

    async def update_state(self, session_id: str, input_text: str) -> dict:
        return {"session_id": session_id, "step": 1}

    async def simulate_world(self, action: str, session_id: str | None = None) -> dict:
        return {
            "risk_score": 0.2,
            "confidence": 0.9,
            "predicted_effects": ["ok"],
            "causal_path": ["ReceiveInput", "PersistAudit"],
        }

    async def verify(
        self,
        patch_ir: str,
        max_loop_iters: int = 2048,
        side_effect_budget: int = 3,
        timeout_ms: int = 80,
    ) -> dict:
        return {"pass": True, "violations": [], "timeout_hit": False, "wasi_ok": True}

    async def model_status(self) -> dict:
        return {"loaded": True, "model_id": "genesis-v0"}

    async def model_infer(self, prompt: str, max_new_tokens: int = 96) -> dict:
        return {
            "model_id": "genesis-v0",
            "text": "Assistant: Da nhan yeu cau va se de xuat patch.",
            "latency_ms": 13,
        }


class FakeKernelFailVerifier(FakeKernelPass):
    async def verify(
        self,
        patch_ir: str,
        max_loop_iters: int = 2048,
        side_effect_budget: int = 3,
        timeout_ms: int = 80,
    ) -> dict:
        return {
            "pass": False,
            "violations": ["infinite_loop_risk"],
            "timeout_hit": False,
            "wasi_ok": True,
        }


class FakeMemoryClient:
    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        return []


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        kernel_url="http://unused",
        host="127.0.0.1",
        port=8080,
        runtime_dir=tmp_path / "runtime",
        dream_hour=2,
        dream_minute=0,
        max_decide_iters=4,
        risk_threshold=0.65,
    )


def test_chat_endpoint_returns_openai_shape_with_policy_metadata(tmp_path: Path) -> None:
    store = EpisodeStore(
        db_path=tmp_path / "episodes.db",
        long_term_path=tmp_path / "memory.jsonl",
    )
    app = create_app(
        settings=_settings(tmp_path),
        kernel_client=FakeKernelPass(),  # type: ignore[arg-type]
        store=store,
        memory_client=FakeMemoryClient(),  # type: ignore[arg-type]
    )
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "vagi-v1",
            "messages": [{"role": "user", "content": "write secure login flow"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert "choices" in body and len(body["choices"]) == 1
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["metadata"]["policy"]["status"] == "pass"
    assert body["metadata"]["policy"]["verifier_pass"] is True
    assert body["metadata"]["model_runtime"]["used"] is True
    assert body["metadata"]["model_runtime"]["model_id"] == "genesis-v0"
    assert body["metadata"]["model_runtime"]["latency_ms"] == 13
    assert "[Kernel: Active | Verifier: Pass | Latency: 13ms]" in body["choices"][0]["message"]["content"]
    client.close()


def test_chat_endpoint_returns_422_when_verifier_gate_fails(tmp_path: Path) -> None:
    store = EpisodeStore(
        db_path=tmp_path / "episodes.db",
        long_term_path=tmp_path / "memory.jsonl",
    )
    app = create_app(
        settings=_settings(tmp_path),
        kernel_client=FakeKernelFailVerifier(),  # type: ignore[arg-type]
        store=store,
        memory_client=FakeMemoryClient(),  # type: ignore[arg-type]
    )
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "vagi-v1",
            "messages": [{"role": "user", "content": "write login flow"}],
            "stream": False,
        },
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "policy_ooda_missing_stage"
    assert any(
        detail["code"] in {"policy_ooda_missing_stage", "policy_verifier_required"}
        for detail in body["error"]["details"]
    )
    client.close()
