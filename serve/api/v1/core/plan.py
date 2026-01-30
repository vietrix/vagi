"""POST /v1/core/plan."""

from __future__ import annotations

from fastapi import APIRouter, Request

from serve.runtime.errors import error_response
from serve.schemas.core import parse_input_ids, parse_obs, parse_task_ids, tensor_to_list

router = APIRouter()


@router.post("/v1/core/plan")
async def plan_core(request: Request):
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
        num_candidates = int(payload.get("numCandidates", 4))
        horizon = int(payload.get("horizon", 3))
        uncertainty_weight = float(payload.get("uncertaintyWeight", 1.0))
        info_gain_weight = float(payload.get("infoGainWeight", 0.0))
        strategy = str(payload.get("strategy", "cem"))

        adapter = request.app.state.adapter
        outputs = adapter.plan(
            input_ids=input_ids,
            obs=obs,
            state=state,
            task_ids=task_ids,
            num_candidates=num_candidates,
            horizon=horizon,
            uncertainty_weight=uncertainty_weight,
            info_gain_weight=info_gain_weight,
            strategy=strategy,
        )

        response = {
            "stateId": state_id,
            "action": tensor_to_list(outputs.get("action")),
            "actionLogits": tensor_to_list(outputs.get("action_logits")),
            "candidateActions": tensor_to_list(outputs.get("candidate_actions")),
            "candidateValues": tensor_to_list(outputs.get("candidate_values")),
            "mode": outputs.get("mode"),
            "earlyStop": outputs.get("early_stop"),
        }
        return response
    except Exception as exc:
        return error_response(str(exc))
