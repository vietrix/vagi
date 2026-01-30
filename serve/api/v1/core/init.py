"""POST /v1/core/init."""

from __future__ import annotations

from fastapi import APIRouter, Request

from serve.runtime.errors import error_response

router = APIRouter()


@router.post("/v1/core/init")
async def init_state(request: Request):
    try:
        payload = await request.json()
        batch_size = int(payload.get("batchSize", 1))
        if batch_size <= 0:
            raise ValueError("batchSize must be > 0")

        adapter = request.app.state.adapter
        store = request.app.state.store
        state = adapter.init_state(batch_size=batch_size)
        stored = store.create(state)
        return {
            "stateId": stored.state_id,
            "timestep": int(stored.state.timestep),
            "model": request.app.state.model_id,
        }
    except Exception as exc:
        return error_response(str(exc))
