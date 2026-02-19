from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, StreamingResponse

from .config import Settings, load_settings
from .dream import DreamScheduler, DreamService
from .errors import PolicyError
from .kernel_client import KernelClient
from .memory import MemoryClient
from .models import (
    ChatChoice,
    ChatChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatUsage,
    DreamRunRequest,
    DreamRunResponse,
    HealthResponse,
    ScanCodeRequest,
    ScanCodeResponse,
)
from .policy import IdentityPolicyEngine
from .reasoning import Reasoner, build_session_id
from .scanner import scan_codebase
from .store import EpisodeStore


@dataclass(slots=True)
class Services:
    settings: Settings
    kernel: KernelClient
    memory: MemoryClient
    store: EpisodeStore
    reasoner: Reasoner
    policy_engine: IdentityPolicyEngine
    dream_service: DreamService
    dream_scheduler: DreamScheduler
    metrics: dict[str, int]


def create_app(
    settings: Settings | None = None,
    kernel_client: KernelClient | None = None,
    store: EpisodeStore | None = None,
    memory_client: MemoryClient | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    kernel_client = kernel_client or KernelClient(base_url=settings.kernel_url)
    memory_client = memory_client or MemoryClient(kernel_url=settings.kernel_url)
    store = store or EpisodeStore(
        db_path=settings.runtime_dir / "episodes.db",
        long_term_path=settings.runtime_dir / "long_term_memory.jsonl",
    )
    reasoner = Reasoner(
        kernel=kernel_client,
        store=store,
        max_decide_iters=settings.max_decide_iters,
        risk_threshold=settings.risk_threshold,
        memory_client=memory_client,
    )
    policy_engine = IdentityPolicyEngine(version="v1")
    dream_service = DreamService(store=store)
    scheduler = DreamScheduler(
        service=dream_service,
        hour=settings.dream_hour,
        minute=settings.dream_minute,
    )
    services = Services(
        settings=settings,
        kernel=kernel_client,
        memory=memory_client,
        store=store,
        reasoner=reasoner,
        policy_engine=policy_engine,
        dream_service=dream_service,
        dream_scheduler=scheduler,
        metrics=defaultdict(int),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        services.dream_scheduler.start()
        yield
        await services.dream_scheduler.stop()
        await services.kernel.close()
        services.store.close()

    app = FastAPI(
        title="vAGI Orchestrator",
        version="0.1.0",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )
    app.state.services = services

    @app.get("/v1/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        kernel_ok = await app.state.services.kernel.healthz()
        return HealthResponse(
            status="ok" if kernel_ok else "degraded",
            kernel_reachable=kernel_ok,
            runtime_dir=str(app.state.services.settings.runtime_dir),
        )

    @app.get("/v1/metrics")
    async def metrics() -> dict[str, Any]:
        base = dict(app.state.services.metrics)
        base.update(app.state.services.store.metrics())
        return base

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> Any:
        services = app.state.services
        services.metrics["chat_requests"] += 1
        seed = "\n".join(msg.content for msg in request.messages)
        session_id = request.session_id or build_session_id(seed)
        messages = [msg.model_dump() for msg in request.messages]

        try:
            services.policy_engine.precheck(messages=messages)
        except PolicyError as exc:
            services.metrics["policy_failures"] += 1
            return ORJSONResponse(status_code=422, content=exc.to_response())

        result: dict[str, Any] | None = None
        try:
            result = await services.reasoner.run_chat(
                session_id=session_id,
                messages=messages,
                runtime_metrics=dict(services.metrics),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if result is None:
            raise HTTPException(status_code=500, detail="reasoner returned empty result")

        verifier_required = bool(result.get("metadata", {}).get("verifier_required", True))
        verifier_pass = bool(result.get("metadata", {}).get("verifier", {}).get("pass", False))
        ooda_trace = dict(result.get("metadata", {}).get("ooda_trace", {}))

        try:
            policy_meta = services.policy_engine.postcheck(result=result)
            services.store.attach_policy_decision(
                episode_id=int(result["episode_id"]),
                policy_pass=True,
                policy_violations=[],
                verifier_required=verifier_required,
                verifier_pass=verifier_pass,
                ooda_trace=ooda_trace,
            )
        except PolicyError as exc:
            services.metrics["policy_failures"] += 1
            if "episode_id" in result:
                services.store.attach_policy_decision(
                    episode_id=int(result["episode_id"]),
                    policy_pass=False,
                    policy_violations=[v.to_dict() for v in exc.violations],
                    verifier_required=verifier_required,
                    verifier_pass=verifier_pass,
                    ooda_trace=ooda_trace,
                )
            return ORJSONResponse(status_code=422, content=exc.to_response())

        if not verifier_pass:
            services.metrics["verifier_failures"] += 1

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:18]}"
        created = int(time.time())
        raw_text = result["content"]
        kernel_latency_ms = int(
            result.get("metadata", {})
            .get("model_runtime", {})
            .get("latency_ms", 0)
        )
        text = _append_runtime_footer(
            raw_text,
            verifier_pass=verifier_pass,
            latency_ms=kernel_latency_ms,
        )
        if request.stream:
            return StreamingResponse(
                _stream_chat_chunks(
                    completion_id=completion_id,
                    model=request.model,
                    text=text,
                    created=created,
                ),
                media_type="text/event-stream",
            )

        prompt_tokens = max(1, sum(len(msg.content.split()) for msg in request.messages))
        completion_tokens = max(1, len(text.split()))
        response = ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatChoiceMessage(content=text),
                )
            ],
            usage=ChatUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            metadata={
                "session_id": session_id,
                "episode_id": result["episode_id"],
                "trust_score": result["trust_score"],
                "policy": policy_meta,
                "safety": {
                    "risk_score": float(
                        result.get("metadata", {}).get("simulation", {}).get("risk_score", 1.0)
                    ),
                    "verifier_pass": verifier_pass,
                },
                "model_runtime": result.get("metadata", {}).get("model_runtime", {}),
            },
        )
        return response

    @app.post("/v1/agents/scan-code", response_model=ScanCodeResponse)
    async def scan_code(request: ScanCodeRequest) -> ScanCodeResponse:
        services = app.state.services
        services.metrics["scan_requests"] += 1
        try:
            scanned_files, issues, remediation_plan = scan_codebase(request.path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return ScanCodeResponse(
            scanned_files=scanned_files,
            issues=issues,
            remediation_plan=remediation_plan,
        )

    @app.post("/v1/evolution/run-dream", response_model=DreamRunResponse)
    async def run_dream(request: DreamRunRequest) -> DreamRunResponse:
        services = app.state.services
        report = await services.dream_service.run_once(source=request.source)
        services.metrics["dream_runs"] += 1
        services.metrics["promoted_episodes"] += int(report["promoted_count"])
        return DreamRunResponse.model_validate(report)

    return app


async def _stream_chat_chunks(
    *,
    completion_id: str,
    model: str,
    text: str,
    created: int,
) -> AsyncGenerator[str, None]:
    chunks = [piece.strip() for piece in text.split("\n") if piece.strip()]
    for piece in chunks:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": piece + "\n"}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0)

    done_payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


app = create_app()


def _append_runtime_footer(content: str, *, verifier_pass: bool, latency_ms: int) -> str:
    verdict = "Pass" if verifier_pass else "Fail"
    footer = f"[Kernel: Active | Verifier: {verdict} | Latency: {latency_ms}ms]"
    body = content.rstrip()
    if body:
        return f"{body}\n{footer}"
    return footer
