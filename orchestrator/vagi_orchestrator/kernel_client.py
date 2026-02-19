from __future__ import annotations

from typing import Any

import httpx


class KernelClient:
    def __init__(self, base_url: str, timeout: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def healthz(self) -> bool:
        try:
            response = await self._client.get("/healthz")
            response.raise_for_status()
            payload = response.json()
            return payload.get("status") == "ok"
        except Exception:
            return False

    async def init_state(self, session_id: str | None = None) -> dict[str, Any]:
        response = await self._client.post(
            "/internal/state/init", json={"session_id": session_id}
        )
        response.raise_for_status()
        return response.json()

    async def update_state(self, session_id: str, input_text: str) -> dict[str, Any]:
        response = await self._client.post(
            "/internal/state/update",
            json={"session_id": session_id, "input": input_text},
        )
        if response.status_code == 404:
            await self.init_state(session_id=session_id)
            response = await self._client.post(
                "/internal/state/update",
                json={"session_id": session_id, "input": input_text},
            )
        response.raise_for_status()
        return response.json()

    async def simulate_world(
        self, action: str, session_id: str | None = None
    ) -> dict[str, Any]:
        response = await self._client.post(
            "/internal/world/simulate",
            json={"session_id": session_id, "action": action},
        )
        response.raise_for_status()
        return response.json()

    async def verify(
        self,
        patch_ir: str,
        max_loop_iters: int = 2048,
        side_effect_budget: int = 3,
        timeout_ms: int = 80,
    ) -> dict[str, Any]:
        response = await self._client.post(
            "/internal/verifier/check",
            json={
                "patch_ir": patch_ir,
                "max_loop_iters": max_loop_iters,
                "side_effect_budget": side_effect_budget,
                "timeout_ms": timeout_ms,
            },
        )
        response.raise_for_status()
        return response.json()

    async def verify_act(
        self,
        patch_ir: str,
        max_steps: int = 16,
        max_output_bytes: int = 8192,
    ) -> dict[str, Any]:
        response = await self._client.post(
            "/internal/verifier/act",
            json={
                "patch_ir": patch_ir,
                "max_steps": max_steps,
                "max_output_bytes": max_output_bytes,
            },
        )
        response.raise_for_status()
        return response.json()

    async def micro_ooda_run(
        self,
        input_text: str,
        session_id: str | None = None,
        risk_threshold: float = 0.45,
        max_decide_iters: int = 2,
    ) -> dict[str, Any]:
        response = await self._client.post(
            "/internal/ooda/micro_run",
            json={
                "session_id": session_id,
                "input": input_text,
                "risk_threshold": risk_threshold,
                "max_decide_iters": max_decide_iters,
            },
        )
        response.raise_for_status()
        return response.json()

    async def homeostasis_status(self) -> dict[str, Any]:
        response = await self._client.get("/internal/homeostasis/status")
        response.raise_for_status()
        return response.json()

    async def homeostasis_event(self, event_type: str, **kwargs: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": event_type}
        payload.update(kwargs)
        response = await self._client.post("/internal/homeostasis/event", json=payload)
        response.raise_for_status()
        return response.json()

    async def model_load(self, model_dir: str) -> dict[str, Any]:
        response = await self._client.post(
            "/internal/model/load",
            json={"model_dir": model_dir},
        )
        response.raise_for_status()
        return response.json()

    async def model_status(self) -> dict[str, Any]:
        response = await self._client.get("/internal/model/status")
        response.raise_for_status()
        return response.json()

    async def model_infer(
        self, prompt: str, max_new_tokens: int = 96
    ) -> dict[str, Any]:
        response = await self._client.post(
            "/internal/infer",
            json={"prompt": prompt, "max_new_tokens": max_new_tokens},
        )
        response.raise_for_status()
        return response.json()

    async def model_infer_mcts(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        num_branches: int | None = None,
        exploration_c: float | None = None,
    ) -> dict[str, Any]:
        """Run MCTS-based inference for multi-branch reasoning."""
        payload: dict[str, Any] = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        }
        if num_branches is not None:
            payload["num_branches"] = num_branches
        if exploration_c is not None:
            payload["exploration_c"] = exploration_c

        response = await self._client.post(
            "/internal/infer/mcts",
            json=payload,
        )
        response.raise_for_status()
        return response.json()
