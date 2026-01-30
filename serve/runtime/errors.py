"""OpenAI-style error helpers."""

from __future__ import annotations

from typing import Any, Optional

from fastapi.responses import JSONResponse


def error_response(
    message: str,
    *,
    status_code: int = 400,
    type_name: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> JSONResponse:
    payload: dict[str, Any] = {
        "error": {
            "message": message,
            "type": type_name,
            "param": param,
            "code": code,
        }
    }
    return JSONResponse(status_code=status_code, content=payload)
