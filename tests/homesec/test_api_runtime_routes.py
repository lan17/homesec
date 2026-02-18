"""Tests for runtime control-plane API routes."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from fastapi.testclient import TestClient

from homesec.api.server import create_app
from homesec.models.config import FastAPIServerConfig
from homesec.runtime.errors import RuntimeReloadConfigError
from homesec.runtime.models import (
    RuntimeReloadRequest,
    RuntimeState,
    RuntimeStatusSnapshot,
)


class _StubRuntimeApp:
    def __init__(
        self,
        *,
        status: RuntimeStatusSnapshot,
        reload_request: RuntimeReloadRequest,
        reload_error: RuntimeReloadConfigError | None = None,
    ) -> None:
        self._status = status
        self._reload_request = reload_request
        self._reload_error = reload_error
        self.wait_calls = 0
        self._config = SimpleNamespace(
            server=FastAPIServerConfig(),
            cameras=[],
        )

    @property
    def config(self):
        return self._config

    def get_runtime_status(self) -> RuntimeStatusSnapshot:
        return self._status

    async def request_runtime_reload(self) -> RuntimeReloadRequest:
        if self._reload_error is not None:
            raise self._reload_error
        return self._reload_request

    async def wait_for_runtime_reload(self) -> None:
        self.wait_calls += 1


def _client(app: _StubRuntimeApp) -> TestClient:
    return TestClient(create_app(app))


def test_runtime_status_endpoint_returns_manager_snapshot() -> None:
    """GET /runtime/status should return runtime state fields."""
    # Given: A runtime snapshot from the manager
    status = RuntimeStatusSnapshot(
        state=RuntimeState.IDLE,
        generation=3,
        reload_in_progress=False,
        active_config_version="abc123",
        last_reload_at=dt.datetime(2026, 2, 13, 4, 0, tzinfo=dt.timezone.utc),
        last_reload_error=None,
    )
    app = _StubRuntimeApp(
        status=status,
        reload_request=RuntimeReloadRequest(accepted=True, message="ok", target_generation=4),
    )
    client = _client(app)

    # When: Requesting runtime status
    response = client.get("/api/v1/runtime/status")

    # Then: Runtime status fields are returned
    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "idle"
    assert payload["generation"] == 3
    assert payload["reload_in_progress"] is False
    assert payload["active_config_version"] == "abc123"
    assert payload["last_reload_error"] is None


def test_runtime_reload_endpoint_returns_success_payload() -> None:
    """POST /runtime/reload should return acceptance payload without waiting."""
    # Given: A successful reload acceptance response from runtime manager
    app = _StubRuntimeApp(
        status=RuntimeStatusSnapshot(
            state=RuntimeState.IDLE,
            generation=1,
            reload_in_progress=False,
            active_config_version="v1",
            last_reload_at=None,
            last_reload_error=None,
        ),
        reload_request=RuntimeReloadRequest(
            accepted=True,
            message="Runtime reload started",
            target_generation=2,
        ),
    )
    client = _client(app)

    # When: Triggering runtime reload
    response = client.post("/api/v1/runtime/reload")

    # Then: Async acceptance is returned immediately
    assert response.status_code == 202
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["message"] == "Runtime reload accepted"
    assert payload["target_generation"] == 2
    assert app.wait_calls == 0


def test_runtime_reload_endpoint_returns_busy_conflict() -> None:
    """POST /runtime/reload should reject when a reload is already in progress."""
    # Given: A reload request rejected by single-flight guard
    app = _StubRuntimeApp(
        status=RuntimeStatusSnapshot(
            state=RuntimeState.RELOADING,
            generation=2,
            reload_in_progress=True,
            active_config_version="v2",
            last_reload_at=None,
            last_reload_error=None,
        ),
        reload_request=RuntimeReloadRequest(
            accepted=False,
            message="Runtime reload already in progress",
            target_generation=3,
        ),
    )
    client = _client(app)

    # When: Triggering runtime reload during in-flight reload
    response = client.post("/api/v1/runtime/reload")

    # Then: A deterministic conflict response is returned
    assert response.status_code == 409
    payload = response.json()
    assert payload["error_code"] == "RELOAD_IN_PROGRESS"
    assert payload["target_generation"] == 3


def test_runtime_reload_endpoint_returns_400_for_invalid_config() -> None:
    """POST /runtime/reload should return 400 for invalid reload config payloads."""
    # Given: A runtime reload request that fails with invalid config
    app = _StubRuntimeApp(
        status=RuntimeStatusSnapshot(
            state=RuntimeState.FAILED,
            generation=2,
            reload_in_progress=False,
            active_config_version="v2",
            last_reload_at=dt.datetime(2026, 2, 13, 4, 0, tzinfo=dt.timezone.utc),
            last_reload_error="invalid config",
        ),
        reload_request=RuntimeReloadRequest(accepted=True, message="unused", target_generation=3),
        reload_error=RuntimeReloadConfigError(
            "Config file not found",
            status_code=400,
            error_code="CONFIG_FILE_NOT_FOUND",
        ),
    )
    client = _client(app)

    # When: Triggering runtime reload
    response = client.post("/api/v1/runtime/reload")

    # Then: A structured 400 error payload is returned
    assert response.status_code == 400
    payload = response.json()
    assert payload["error_code"] == "CONFIG_FILE_NOT_FOUND"
    assert payload["detail"] == "Config file not found"


def test_runtime_reload_endpoint_returns_422_for_unprocessable_config() -> None:
    """POST /runtime/reload should return 422 for semantically invalid config."""
    # Given: A runtime reload request that fails config validation
    app = _StubRuntimeApp(
        status=RuntimeStatusSnapshot(
            state=RuntimeState.FAILED,
            generation=2,
            reload_in_progress=False,
            active_config_version="v2",
            last_reload_at=dt.datetime(2026, 2, 13, 4, 0, tzinfo=dt.timezone.utc),
            last_reload_error="unprocessable config",
        ),
        reload_request=RuntimeReloadRequest(accepted=True, message="unused", target_generation=3),
        reload_error=RuntimeReloadConfigError(
            "Config validation failed",
            status_code=422,
            error_code="CONFIG_VALIDATION_FAILED",
        ),
    )
    client = _client(app)

    # When: Triggering runtime reload
    response = client.post("/api/v1/runtime/reload")

    # Then: A structured 422 error payload is returned
    assert response.status_code == 422
    payload = response.json()
    assert payload["error_code"] == "CONFIG_VALIDATION_FAILED"
    assert payload["detail"] == "Config validation failed"
