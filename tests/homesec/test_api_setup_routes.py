"""Tests for setup API routes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from homesec.api.server import create_app
from homesec.config.loader import ConfigError, ConfigErrorCode
from homesec.config.manager import ConfigManager
from homesec.models.config import FastAPIServerConfig
from homesec.models.setup import (
    FinalizeResponse,
    PreflightCheckResponse,
    PreflightResponse,
    SetupStatusResponse,
)
from homesec.models.setup import (
    TestConnectionResponse as SetupTestConnectionResponse,
)
from homesec.services import setup as setup_service


class _StubSetupRepository:
    async def ping(self) -> bool:
        return True


class _StubSetupStorage:
    async def ping(self) -> bool:
        return True


class _StubSetupApp:
    def __init__(
        self,
        *,
        bootstrap_mode: bool,
        server_config: FastAPIServerConfig,
        config_manager: ConfigManager | None = None,
    ) -> None:
        self.bootstrap_mode = bootstrap_mode
        self.repository = _StubSetupRepository()
        self.storage = _StubSetupStorage()
        self.config_manager = config_manager or ConfigManager(Path("config/config.yaml"))
        self.sources: list[object] = []
        self.pipeline_running = False
        self.uptime_seconds = 0.0
        self._config = SimpleNamespace(
            server=server_config,
            cameras=[],
        )

    @property
    def config(self):  # type: ignore[override]
        if self.bootstrap_mode:
            raise RuntimeError("Config not loaded")
        return self._config

    @property
    def server_config(self) -> FastAPIServerConfig:
        return self._config.server

    def get_source(self, camera_name: str) -> None:
        _ = camera_name
        return None

    def request_restart(self) -> None:
        return None


def _client(app: _StubSetupApp) -> TestClient:
    return TestClient(create_app(app))


def test_setup_status_route_requires_api_key_when_auth_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setup status route should require API key when auth is enabled."""
    # Given: Auth-enabled server config and setup status service response stub
    monkeypatch.setenv("HOMESEC_TEST_API_KEY", "secret")
    app = _StubSetupApp(
        bootstrap_mode=False,
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_TEST_API_KEY"),
    )
    client = _client(app)

    # When: Calling setup status without Authorization header
    response = client.get("/api/v1/setup/status")

    # Then: Endpoint is rejected by API-key auth dependency
    assert response.status_code == 401
    payload = response.json()
    assert payload["error_code"] == "UNAUTHORIZED"


def test_setup_status_route_accepts_api_key_when_auth_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setup status route should allow requests with valid API key."""
    # Given: Auth-enabled server config and valid API key header
    monkeypatch.setenv("HOMESEC_TEST_API_KEY", "secret")
    app = _StubSetupApp(
        bootstrap_mode=False,
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_TEST_API_KEY"),
    )
    client = _client(app)

    # When: Calling setup status with Authorization header
    response = client.get("/api/v1/setup/status", headers={"Authorization": "Bearer secret"})

    # Then: Endpoint remains available with auth
    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] in {"fresh", "partial", "complete"}
    assert "has_cameras" in payload


def test_setup_status_route_reports_fresh_in_bootstrap_mode() -> None:
    """Setup status route should report fresh state in bootstrap mode."""
    # Given: A bootstrap-mode app without loaded config
    app = _StubSetupApp(
        bootstrap_mode=True,
        server_config=FastAPIServerConfig(auth_enabled=False),
    )
    client = _client(app)

    # When: Calling setup status route
    response = client.get("/api/v1/setup/status")

    # Then: Endpoint reports fresh setup state
    assert response.status_code == 200
    assert response.json()["state"] == "fresh"


def test_setup_preflight_route_returns_service_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setup preflight route should return service-layer response model unchanged."""
    # Given: A deterministic preflight response from the setup service layer
    app = _StubSetupApp(
        bootstrap_mode=False,
        server_config=FastAPIServerConfig(auth_enabled=False),
    )
    client = _client(app)

    async def _fake_run_preflight(_: object) -> PreflightResponse:
        return PreflightResponse(
            checks=[
                PreflightCheckResponse(
                    name="postgres",
                    passed=True,
                    message="Database reachable",
                    latency_ms=1.0,
                )
            ],
            all_passed=True,
        )

    monkeypatch.setattr("homesec.api.routes.setup.run_preflight", _fake_run_preflight)

    # When: Calling setup preflight route
    response = client.post("/api/v1/setup/preflight")

    # Then: Route returns service payload without auth requirements
    assert response.status_code == 200
    payload = response.json()
    assert payload["all_passed"] is True
    assert payload["checks"][0]["name"] == "postgres"


def test_setup_test_connection_route_delegates_to_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setup test-connection route should delegate and return service outcome."""
    # Given: A deterministic test-connection response from the setup service layer
    app = _StubSetupApp(
        bootstrap_mode=False,
        server_config=FastAPIServerConfig(auth_enabled=False),
    )
    client = _client(app)

    async def _fake_test_connection(_: object, __: object) -> SetupTestConnectionResponse:
        return SetupTestConnectionResponse(
            success=True,
            message="Probe passed",
            latency_ms=8.5,
            details={"backend": "mqtt"},
        )

    monkeypatch.setattr("homesec.api.routes.setup.test_connection", _fake_test_connection)

    # When: Calling setup test-connection route
    response = client.post(
        "/api/v1/setup/test-connection",
        json={"type": "notifier", "backend": "mqtt", "config": {"host": "localhost"}},
    )

    # Then: Route returns service payload unchanged
    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "message": "Probe passed",
        "latency_ms": 8.5,
        "details": {"backend": "mqtt"},
    }


def test_setup_test_connection_route_maps_request_error_to_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setup test-connection route should map request dispatch errors to canonical 400."""
    # Given: A setup service request error with available backend hints
    app = _StubSetupApp(
        bootstrap_mode=False,
        server_config=FastAPIServerConfig(auth_enabled=False),
    )
    client = _client(app)

    async def _fake_test_connection(_: object, __: object) -> SetupTestConnectionResponse:
        raise setup_service.SetupTestConnectionRequestError(
            "Unknown notifier backend 'foo'.",
            available_backends=["mqtt", "sendgrid_email"],
        )

    monkeypatch.setattr("homesec.api.routes.setup.test_connection", _fake_test_connection)

    # When: Calling setup test-connection route with unknown backend
    response = client.post(
        "/api/v1/setup/test-connection",
        json={"type": "notifier", "backend": "foo", "config": {}},
    )

    # Then: Route emits canonical BAD_REQUEST envelope with backend hints
    assert response.status_code == 400
    payload = response.json()
    assert payload["error_code"] == "BAD_REQUEST"
    assert payload["available_backends"] == ["mqtt", "sendgrid_email"]


def test_setup_test_connection_route_returns_canonical_422_for_invalid_payload() -> None:
    """Setup test-connection route should return canonical validation envelope for bad input."""
    # Given: A setup app and a malformed test-connection payload with invalid target type
    app = _StubSetupApp(
        bootstrap_mode=False,
        server_config=FastAPIServerConfig(auth_enabled=False),
    )
    client = _client(app)

    # When: Calling setup test-connection with an invalid request body
    response = client.post(
        "/api/v1/setup/test-connection",
        json={"type": "invalid-target", "backend": "mqtt", "config": {}},
    )

    # Then: Route returns canonical request-validation error envelope
    assert response.status_code == 422
    payload = response.json()
    assert payload["detail"] == "Request validation failed"
    assert payload["error_code"] == "REQUEST_VALIDATION_FAILED"
    assert isinstance(payload["validation_errors"], list)
    assert payload["validation_errors"]


def test_setup_status_route_delegates_to_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setup status route should delegate response generation to setup service."""
    # Given: A deterministic setup status result from service layer
    app = _StubSetupApp(
        bootstrap_mode=False,
        server_config=FastAPIServerConfig(auth_enabled=False),
    )
    client = _client(app)

    async def _fake_get_setup_status(_: object) -> SetupStatusResponse:
        return SetupStatusResponse(
            state="partial",
            has_cameras=True,
            pipeline_running=False,
            auth_configured=True,
        )

    monkeypatch.setattr("homesec.api.routes.setup.get_setup_status", _fake_get_setup_status)

    # When: Calling setup status route
    response = client.get("/api/v1/setup/status")

    # Then: Route returns service response payload
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "state": "partial",
        "has_cameras": True,
        "pipeline_running": False,
        "auth_configured": True,
    }


def test_setup_finalize_requires_bootstrap_mode() -> None:
    """Finalize route should reject requests when app is not in bootstrap mode."""
    # Given: A non-bootstrap app
    app = _StubSetupApp(
        bootstrap_mode=False,
        server_config=FastAPIServerConfig(auth_enabled=False),
    )
    client = _client(app)

    # When: Calling setup finalize
    response = client.post("/api/v1/setup/finalize", json={})

    # Then: Route rejects finalize outside setup mode
    assert response.status_code == 409
    payload = response.json()
    assert payload["error_code"] == "CONFLICT"


def test_setup_finalize_delegates_to_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finalize route should delegate orchestration to setup service."""
    # Given: A bootstrap app and deterministic finalize response
    app = _StubSetupApp(
        bootstrap_mode=True,
        server_config=FastAPIServerConfig(auth_enabled=False),
    )
    client = _client(app)

    async def _fake_finalize_setup(_: object, __: object) -> FinalizeResponse:
        return FinalizeResponse(
            success=True,
            config_path="/tmp/config.yaml",
            restart_requested=True,
            defaults_applied=["storage"],
            errors=[],
        )

    monkeypatch.setattr("homesec.api.routes.setup.finalize_setup", _fake_finalize_setup)

    # When: Calling setup finalize
    response = client.post("/api/v1/setup/finalize", json={})

    # Then: Route returns service payload
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "success": True,
        "config_path": "/tmp/config.yaml",
        "restart_requested": True,
        "defaults_applied": ["storage"],
        "errors": [],
    }


def test_setup_finalize_returns_canonical_422_for_semantic_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalize route should map semantic validation errors to canonical 422 envelope."""
    # Given: A bootstrap app and service-layer semantic validation failure
    app = _StubSetupApp(
        bootstrap_mode=True,
        server_config=FastAPIServerConfig(auth_enabled=False),
    )
    client = _client(app)

    async def _fake_finalize_setup(_: object, __: object) -> FinalizeResponse:
        raise setup_service.SetupFinalizeValidationError(
            ["At least one camera must be configured before finalizing setup."]
        )

    monkeypatch.setattr("homesec.api.routes.setup.finalize_setup", _fake_finalize_setup)

    # When: Calling setup finalize
    response = client.post("/api/v1/setup/finalize", json={})

    # Then: Route returns canonical 422 with setup-specific error code
    assert response.status_code == 422
    payload = response.json()
    assert payload["error_code"] == "SETUP_FINALIZE_INVALID"
    assert payload["errors"]


def test_setup_finalize_returns_canonical_422_for_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalize route should map config-loader errors to canonical 422 envelope."""
    # Given: A bootstrap app and service-layer config validation error
    app = _StubSetupApp(
        bootstrap_mode=True,
        server_config=FastAPIServerConfig(auth_enabled=False),
    )
    client = _client(app)

    async def _fake_finalize_setup(_: object, __: object) -> FinalizeResponse:
        raise ConfigError(
            "Config validation failed",
            code=ConfigErrorCode.VALIDATION_FAILED,
        )

    monkeypatch.setattr("homesec.api.routes.setup.finalize_setup", _fake_finalize_setup)

    # When: Calling setup finalize
    response = client.post("/api/v1/setup/finalize", json={})

    # Then: Route returns canonical 422 with propagated config error code
    assert response.status_code == 422
    payload = response.json()
    assert payload["error_code"] == ConfigErrorCode.VALIDATION_FAILED.value
    assert payload["detail"] == "Config validation failed"
