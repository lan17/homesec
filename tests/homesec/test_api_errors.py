"""Tests for canonical API error handlers."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from homesec.api.dependencies import get_homesec_app
from homesec.api.errors import register_exception_handlers


def test_http_exception_dict_detail_is_normalized_to_canonical_envelope() -> None:
    """HTTPException payloads with dict detail should map to canonical envelope."""
    # Given: An app with canonical error handlers and a route raising dict-detail HTTPException
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    async def _boom() -> None:
        raise HTTPException(
            status_code=409,
            detail={
                "detail": "reload busy",
                "error_code": "RELOAD_IN_PROGRESS",
                "target_generation": 7,
            },
        )

    client = TestClient(app, raise_server_exceptions=False)

    # When: Calling the route
    response = client.get("/boom")

    # Then: Canonical envelope and extra fields are preserved
    assert response.status_code == 409
    payload = response.json()
    assert payload["detail"] == "reload busy"
    assert payload["error_code"] == "RELOAD_IN_PROGRESS"
    assert payload["target_generation"] == 7


def test_unhandled_exception_maps_to_internal_server_error_envelope() -> None:
    """Unhandled route exceptions should return canonical internal server error."""
    # Given: An app with canonical error handlers and a route raising an unhandled exception
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/crash")
    async def _crash() -> None:
        raise RuntimeError("unexpected failure")

    client = TestClient(app, raise_server_exceptions=False)

    # When: Calling the crashing route
    response = client.get("/crash")

    # Then: Internal server error envelope is returned
    assert response.status_code == 500
    payload = response.json()
    assert payload["detail"] == "Internal server error"
    assert payload["error_code"] == "INTERNAL_SERVER_ERROR"


def test_missing_homesec_app_maps_to_app_not_initialized_error() -> None:
    """Dependency failure for missing app state should map to canonical 503 payload."""
    # Given: An app with canonical error handlers and a route depending on HomeSec app state
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/needs-app")
    async def _needs_app(app_instance: object = Depends(get_homesec_app)) -> None:
        _ = app_instance
        return None

    client = TestClient(app)

    # When: Calling the route without app.state.homesec configured
    response = client.get("/needs-app")

    # Then: Canonical app-not-initialized response is returned
    assert response.status_code == 503
    payload = response.json()
    assert payload["detail"] == "Application not initialized"
    assert payload["error_code"] == "APP_NOT_INITIALIZED"
