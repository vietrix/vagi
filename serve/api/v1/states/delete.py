"""POST /v1/states/delete."""

from __future__ import annotations

from fastapi import APIRouter, Request

from serve.runtime.errors import error_response

router = APIRouter()


@router.post("/v1/states/delete")
async def delete_state(request: Request):
    try:
        payload = await request.json()
        state_id = payload.get("stateId")
        if not state_id:
            raise ValueError("stateId is required")
        store = request.app.state.store
        deleted = store.delete(state_id)
        if not deleted:
            return error_response("stateId not found", status_code=404)
        return {"stateId": state_id, "deleted": True}
    except Exception as exc:
        return error_response(str(exc))
