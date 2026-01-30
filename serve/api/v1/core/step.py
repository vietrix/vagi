"""POST /v1/core/step."""

from __future__ import annotations

from fastapi import APIRouter, Request

from serve.runtime.errors import error_response
from serve.schemas.core import outputs_to_payload, parse_input_ids, parse_obs, parse_task_ids

router = APIRouter()


@router.post("/v1/core/step")
async def step_core(request: Request):
    try:
        payload = await request.json()
        state_id = payload.get("stateId")
        if not state_id:
            raise ValueError("stateId is required")

        store = request.app.state.store
        state = store.get(state_id)
        if state is None:
            return error_response("stateId not found", status_code=404)

        input_ids = parse_input_ids(payload)
        obs = parse_obs(payload, required=True)
        task_ids = parse_task_ids(payload)

        adapter = request.app.state.adapter
        outputs = adapter.step(input_ids=input_ids, obs=obs, state=state, task_ids=task_ids)
        store.set(state_id, outputs["state"])

        response = outputs_to_payload(outputs)
        response.update({"stateId": state_id, "timestep": int(outputs["state"].timestep)})
        return response
    except Exception as exc:
        return error_response(str(exc))
