"""Tests for setup API routes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from homesec.api.server import create_app
from homesec.models.config import FastAPIServerConfig
from homesec.models.setup import PreflightCheckResponse, PreflightResponse, SetupStatusResponse


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
    ) -> None:
        self.bootstrap_mode = bootstrap_mode
        self.repository = _StubSetupRepository()
        self.storage = _StubSetupStorage()
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


def _client(app: _StubSetupApp) -> TestClient:
    return TestClient(create_app(app))


def test_setup_status_route_is_public_when_auth_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setup status route should remain public even when API key auth is enabled."""
    # Given: Auth-enabled server config and setup status service response stub
    monkeypatch.setenv("HOMESEC_TEST_API_KEY", "secret")
    app = _StubSetupApp(
        bootstrap_mode=False,
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_TEST_API_KEY"),
    )
    client = _client(app)

    # When: Calling setup status without Authorization header
    response = client.get("/api/v1/setup/status")

    # Then: Endpoint is accessible without auth and returns setup payload
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
