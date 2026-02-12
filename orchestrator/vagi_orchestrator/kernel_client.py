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

    async def weave_plan(
        self,
        *,
        query: str,
        input_value: int,
        bindings: dict[str, str] | None = None,
        top_k: int = 3,
        risk_threshold: float = 0.65,
        verifier_required: bool = True,
        session_id: str | None = None,
    ) -> dict[str, Any] | None:
        response = await self._client.post(
            "/internal/hdc/weave/plan",
            json={
                "query": query,
                "input": input_value,
                "top_k": max(1, min(top_k, 10)),
                "risk_threshold": risk_threshold,
                "verifier_required": verifier_required,
                "session_id": session_id,
                "bindings": bindings or {},
            },
        )
        if response.status_code in {400, 404, 422}:
            return None
        response.raise_for_status()
        return response.json()

    async def evolve_templates(
        self,
        *,
        template_id: str | None = None,
        query: str | None = None,
        generations: int = 3,
        population_size: int = 8,
        survivors: int = 2,
        risk_threshold: float = 0.65,
        seed_input: int = 13,
        promote: bool = True,
    ) -> dict[str, Any] | None:
        population_size = max(2, min(population_size, 64))
        survivors = max(1, min(survivors, population_size))
        response = await self._client.post(
            "/internal/hdc/evolution/mutate",
            json={
                "template_id": template_id,
                "query": query,
                "generations": max(1, min(generations, 20)),
                "population_size": population_size,
                "survivors": survivors,
                "risk_threshold": risk_threshold,
                "seed_input": seed_input,
                "promote": promote,
            },
        )
        if response.status_code in {400, 404, 422}:
            return None
        response.raise_for_status()
        return response.json()
