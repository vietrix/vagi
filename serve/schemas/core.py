"""Request/response helpers for core endpoints."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def _require(payload: Dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"Missing required field: {key}")
    return payload[key]


def parse_input_ids(payload: Dict[str, Any]) -> torch.Tensor:
    input_ids = _require(payload, "inputIds")
    return torch.tensor(input_ids, dtype=torch.long)


def parse_obs(payload: Dict[str, Any], *, required: bool = False) -> Optional[torch.Tensor]:
    if "obs" not in payload:
        if required:
            raise ValueError("Missing required field: obs")
        return None
    return torch.tensor(payload["obs"], dtype=torch.float32)


def parse_task_ids(payload: Dict[str, Any]) -> Optional[torch.Tensor]:
    if "taskIds" not in payload:
        return None
    return torch.tensor(payload["taskIds"], dtype=torch.long)


def tensor_to_list(value: Optional[torch.Tensor]) -> Optional[list]:
    if value is None:
        return None
    return value.detach().cpu().tolist()


def budget_to_payload(budget: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if budget is None:
        return None
    return {
        "mode": budget.get("mode"),
        "horizon": budget.get("horizon"),
        "numCandidates": budget.get("numCandidates", budget.get("num_candidates")),
    }


def outputs_to_payload(outputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "textLogits": tensor_to_list(outputs.get("text_logits")),
        "actionLogits": tensor_to_list(outputs.get("action_logits")),
        "value": tensor_to_list(outputs.get("value")),
        "valueLogvar": tensor_to_list(outputs.get("value_logvar")),
        "worldPred": tensor_to_list(outputs.get("world_pred")),
        "worldLogvar": tensor_to_list(outputs.get("world_logvar")),
        "errorLogits": tensor_to_list(outputs.get("error_logits")),
        "infoGain": tensor_to_list(outputs.get("info_gain")),
        "budgetModeLogits": tensor_to_list(outputs.get("budget_mode_logits")),
        "budgetHorizonLogits": tensor_to_list(outputs.get("budget_horizon_logits")),
        "budgetCandidateLogits": tensor_to_list(outputs.get("budget_candidate_logits")),
        "confidence": tensor_to_list(outputs.get("confidence")),
        "uncertainty": tensor_to_list(outputs.get("uncertainty")),
        "budget": budget_to_payload(outputs.get("budget")),
        "stopReason": outputs.get("stopReason"),
    }
