"""POST /v1/states/reset."""

from __future__ import annotations

from fastapi import APIRouter, Request

from serve.runtime.errors import error_response

router = APIRouter()


@router.post("/v1/states/reset")
async def reset_state(request: Request):
    try:
        payload = await request.json()
        state_id = payload.get("stateId")
        if not state_id:
            raise ValueError("stateId is required")
        batch_size = int(payload.get("batchSize", 1))
        if batch_size <= 0:
            raise ValueError("batchSize must be > 0")

        adapter = request.app.state.adapter
        store = request.app.state.store
        state = adapter.init_state(batch_size=batch_size)
        store.reset(state_id, state)
        return {"stateId": state_id, "timestep": int(state.timestep)}
    except Exception as exc:
        return error_response(str(exc))
