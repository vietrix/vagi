import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI

from serve.app.main import createApp


def test_create_app_routes() -> None:
    app = createApp()
    assert isinstance(app, FastAPI)
    assert hasattr(app.state, "adapter")
    assert hasattr(app.state, "store")
    assert hasattr(app.state, "model_id")

    paths = {route.path for route in app.routes}
    for path in [
        "/v1/models",
        "/v1/core/init",
        "/v1/core/step",
        "/v1/core/plan",
        "/v1/states/reset",
        "/v1/states/delete",
    ]:
        assert path in paths
