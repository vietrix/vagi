"""FastAPI entrypoint for vAGI serving."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from serve.app.config import load_config
from serve.runtime.adapters.core import CoreAdapter
from serve.runtime.state.store import StateStore
from serve.api.v1.models.get import router as models_router
from serve.api.v1.core.init import router as core_init_router
from serve.api.v1.core.step import router as core_step_router
from serve.api.v1.core.plan import router as core_plan_router
from serve.api.v1.states.reset import router as states_reset_router
from serve.api.v1.states.delete import router as states_delete_router


def createApp() -> FastAPI:
    cfg = load_config()
    app = FastAPI(title="vAGI API", version="1.0")

    adapter = CoreAdapter(cfg.build_model_config(), device=cfg.device)
    store = StateStore()

    app.state.model_id = cfg.model_id
    app.state.adapter = adapter
    app.state.store = store

    app.include_router(models_router)
    app.include_router(core_init_router)
    app.include_router(core_step_router)
    app.include_router(core_plan_router)
    app.include_router(states_reset_router)
    app.include_router(states_delete_router)
    ui_dir = Path(__file__).resolve().parents[1] / "ui"
    app.mount("/ui", StaticFiles(directory=ui_dir.as_posix(), html=True), name="ui")
    return app
