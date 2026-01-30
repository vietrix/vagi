"""POST /v1/core/plan."""

from __future__ import annotations

from fastapi import APIRouter, Request

from serve.runtime.errors import error_response
from serve.schemas.core import budget_to_payload, parse_input_ids, parse_obs, parse_task_ids, tensor_to_list

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
        max_horizon = payload.get("maxHorizon")
        max_candidates = payload.get("maxCandidates")
        max_steps = payload.get("maxSteps")
        risk_penalty = payload.get("riskPenalty")
        min_confidence = payload.get("minConfidenceToAct")
        policy_only = bool(payload.get("policyOnly", False))
        trace = bool(payload.get("trace", False))
        action_validity_threshold = payload.get("actionValidityThreshold")
        ood_uncertainty = payload.get("oodUncertaintyThreshold")
        ood_trace = payload.get("oodTraceThreshold")
        ood_policy = payload.get("oodPolicy")
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
            max_horizon=int(max_horizon) if max_horizon is not None else None,
            max_candidates=int(max_candidates) if max_candidates is not None else None,
            max_steps=int(max_steps) if max_steps is not None else None,
            risk_penalty=float(risk_penalty) if risk_penalty is not None else None,
            min_confidence_to_act=float(min_confidence) if min_confidence is not None else None,
            policy_only=policy_only,
            trace=trace,
            action_validity_threshold=float(action_validity_threshold)
            if action_validity_threshold is not None
            else None,
            ood_uncertainty_threshold=float(ood_uncertainty) if ood_uncertainty is not None else None,
            ood_trace_threshold=float(ood_trace) if ood_trace is not None else None,
            ood_policy=str(ood_policy) if ood_policy is not None else None,
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
            "confidence": tensor_to_list(outputs.get("confidence")),
            "uncertainty": tensor_to_list(outputs.get("uncertainty")),
            "budget": budget_to_payload(outputs.get("budget")),
            "stopReason": outputs.get("stopReason"),
            "trace": outputs.get("trace"),
        }
        return response
    except Exception as exc:
        return error_response(str(exc))
