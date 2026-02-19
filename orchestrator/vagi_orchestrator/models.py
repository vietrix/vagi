from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "vagi-v1"
    messages: list[Message] = Field(min_length=1)
    stream: bool = False
    session_id: str | None = None


class ChatChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatChoice(BaseModel):
    index: int
    message: ChatChoiceMessage
    finish_reason: Literal["stop"] = "stop"


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage
    metadata: dict[str, Any]


class ScanCodeRequest(BaseModel):
    path: str


class ScanIssue(BaseModel):
    path: str
    line: int
    severity: Literal["low", "medium", "high"]
    rule: str
    message: str


class ScanCodeResponse(BaseModel):
    scanned_files: int
    issues: list[ScanIssue]
    remediation_plan: list[str]


class DreamRunRequest(BaseModel):
    source: str = "manual"


class DreamRunResponse(BaseModel):
    run_id: str
    source: str
    promoted_count: int
    pass_rate: float
    regression_fail: int
    threshold: float
    promoted_episode_ids: list[int]
    self_corrected: int = 0


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    kernel_reachable: bool
    runtime_dir: str
