"""GET /v1/models."""

from __future__ import annotations

from fastapi import APIRouter, Request

from serve.runtime.errors import error_response

router = APIRouter()


@router.get("/v1/models")
def get_models(request: Request):
    try:
        model_id = request.app.state.model_id
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}
    except Exception as exc:  # pragma: no cover
        return error_response(str(exc), status_code=500, type_name="internal_error")
