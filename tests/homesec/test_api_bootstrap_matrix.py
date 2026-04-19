"""Cross-group bootstrap/dependency behavior matrix tests."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from fastapi.testclient import TestClient

from homesec.models.clip import ClipStateData
from homesec.models.config import FastAPIServerConfig
from homesec.models.enums import ClipStatus
from homesec.runtime.models import RuntimeState, RuntimeStatusSnapshot
from homesec.services import setup as setup_service
from tests.homesec.test_api_routes import (
    _client,
    _StubApp,
    _StubRepository,
    _StubSource,
    _StubStorage,
    _write_config,
)


class _MatrixStubApp(_StubApp):
    """Extends the API route stub app with runtime status access."""

    def __init__(
        self,
        *,
        runtime_status: RuntimeStatusSnapshot,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._runtime_status = runtime_status

    def get_runtime_status(self) -> RuntimeStatusSnapshot:
        return self._runtime_status


@dataclass(frozen=True)
class _MatrixCase:
    name: str
    method: Literal["GET", "POST"]
    path: str
    auth_enabled: bool
    db_ok: bool
    pipeline_running: bool
    auth_header: str | None
    include_clip: bool
    expected_status: int
    bootstrap_mode: bool = False
    expected_error_code: str | None = None
    request_json: dict[str, object] | None = None


def _sample_clip() -> ClipStateData:
    return ClipStateData(
        clip_id="clip-1",
        camera_name="front",
        status=ClipStatus.UPLOADED,
        local_path="/tmp/clip-1.mp4",
        storage_uri="dropbox:/clips/clip-1.mp4",
        created_at=dt.datetime(2026, 2, 19, tzinfo=dt.timezone.utc),
    )


def _build_client(tmp_path: Path, case: _MatrixCase) -> tuple[TestClient, _StubRepository]:
    server_config = FastAPIServerConfig(
        auth_enabled=case.auth_enabled, api_key_env="HOMESEC_API_KEY"
    )
    manager = _write_config(
        tmp_path,
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp/front"}},
            }
        ],
    )
    repository = _StubRepository(clips=[_sample_clip()] if case.include_clip else [], ok=case.db_ok)
    app = _MatrixStubApp(
        runtime_status=RuntimeStatusSnapshot(
            state=RuntimeState.IDLE,
            generation=1,
            reload_in_progress=False,
            active_config_version="v1",
            last_reload_at=None,
            last_reload_error=None,
        ),
        config_manager=manager,
        repository=repository,
        storage=_StubStorage(media_bytes=b"video"),
        sources_by_name={"front": _StubSource(healthy=True, heartbeat=1.0)},
        server_config=server_config,
        pipeline_running=case.pipeline_running,
        bootstrap_mode=case.bootstrap_mode,
    )
    app.uptime_seconds = 12.0
    return _client(app), repository


def _send_request(client: TestClient, case: _MatrixCase, headers: dict[str, str]):
    if case.method == "GET":
        return client.get(case.path, headers=headers)
    request_json = case.request_json if case.request_json is not None else {}
    return client.post(case.path, headers=headers, json=request_json)


@pytest.mark.parametrize(
    ("case"),
    [
        _MatrixCase(
            name="public_health_versioned_stays_public_with_auth_enabled",
            method="GET",
            path="/api/v1/health",
            auth_enabled=True,
            db_ok=True,
            pipeline_running=True,
            auth_header=None,
            include_clip=False,
            expected_status=200,
        ),
        _MatrixCase(
            name="public_health_root_stays_public_with_auth_enabled",
            method="GET",
            path="/health",
            auth_enabled=True,
            db_ok=True,
            pipeline_running=True,
            auth_header=None,
            include_clip=False,
            expected_status=200,
        ),
        _MatrixCase(
            name="public_health_degrades_when_db_is_down",
            method="GET",
            path="/api/v1/health",
            auth_enabled=True,
            db_ok=False,
            pipeline_running=True,
            auth_header=None,
            include_clip=False,
            expected_status=200,
        ),
        _MatrixCase(
            name="public_health_returns_503_when_pipeline_stopped",
            method="GET",
            path="/api/v1/health",
            auth_enabled=False,
            db_ok=True,
            pipeline_running=False,
            auth_header=None,
            include_clip=False,
            expected_status=503,
        ),
        _MatrixCase(
            name="public_health_reports_setup_required_in_bootstrap_mode",
            method="GET",
            path="/api/v1/health",
            auth_enabled=False,
            db_ok=True,
            pipeline_running=False,
            auth_header=None,
            include_clip=False,
            expected_status=200,
            bootstrap_mode=True,
        ),
        _MatrixCase(
            name="setup_status_requires_api_key_when_auth_enabled",
            method="GET",
            path="/api/v1/setup/status",
            auth_enabled=True,
            db_ok=False,
            pipeline_running=False,
            auth_header=None,
            include_clip=False,
            expected_status=401,
            expected_error_code="UNAUTHORIZED",
        ),
        _MatrixCase(
            name="setup_preflight_requires_api_key_when_auth_enabled",
            method="POST",
            path="/api/v1/setup/preflight",
            auth_enabled=True,
            db_ok=False,
            pipeline_running=False,
            auth_header=None,
            include_clip=False,
            expected_status=401,
            expected_error_code="UNAUTHORIZED",
        ),
        _MatrixCase(
            name="setup_test_connection_requires_api_key_when_auth_enabled",
            method="POST",
            path="/api/v1/setup/test-connection",
            auth_enabled=True,
            db_ok=False,
            pipeline_running=False,
            auth_header=None,
            include_clip=False,
            expected_status=401,
            expected_error_code="UNAUTHORIZED",
            request_json={"type": "storage", "backend": "local", "config": {"root": "/tmp"}},
        ),
        _MatrixCase(
            name="setup_test_connection_accepts_api_key_when_auth_enabled",
            method="POST",
            path="/api/v1/setup/test-connection",
            auth_enabled=True,
            db_ok=False,
            pipeline_running=False,
            auth_header="Bearer secret",
            include_clip=False,
            expected_status=200,
            request_json={"type": "storage", "backend": "local", "config": {"root": "/tmp"}},
        ),
        _MatrixCase(
            name="setup_status_remains_available_in_bootstrap_mode",
            method="GET",
            path="/api/v1/setup/status",
            auth_enabled=False,
            db_ok=False,
            pipeline_running=False,
            auth_header=None,
            include_clip=False,
            expected_status=200,
            bootstrap_mode=True,
        ),
        _MatrixCase(
            name="setup_test_connection_remains_available_in_bootstrap_mode",
            method="POST",
            path="/api/v1/setup/test-connection",
            auth_enabled=False,
            db_ok=False,
            pipeline_running=False,
            auth_header=None,
            include_clip=False,
            expected_status=200,
            bootstrap_mode=True,
            request_json={"type": "storage", "backend": "local", "config": {"root": "/tmp"}},
        ),
        _MatrixCase(
            name="setup_preflight_remains_available_in_bootstrap_mode",
            method="POST",
            path="/api/v1/setup/preflight",
            auth_enabled=False,
            db_ok=False,
            pipeline_running=False,
            auth_header=None,
            include_clip=False,
            expected_status=200,
            bootstrap_mode=True,
        ),
        _MatrixCase(
            name="setup_finalize_requires_bootstrap_mode",
            method="POST",
            path="/api/v1/setup/finalize",
            auth_enabled=False,
            db_ok=False,
            pipeline_running=False,
            auth_header=None,
            include_clip=False,
            expected_status=409,
            expected_error_code="CONFLICT",
        ),
        _MatrixCase(
            name="setup_finalize_remains_available_in_bootstrap_mode",
            method="POST",
            path="/api/v1/setup/finalize",
            auth_enabled=False,
            db_ok=False,
            pipeline_running=False,
            auth_header=None,
            include_clip=False,
            expected_status=200,
            bootstrap_mode=True,
        ),
        _MatrixCase(
            name="config_requires_api_key_when_auth_enabled",
            method="GET",
            path="/api/v1/config",
            auth_enabled=True,
            db_ok=True,
            pipeline_running=True,
            auth_header=None,
            include_clip=False,
            expected_status=401,
            expected_error_code="UNAUTHORIZED",
        ),
        _MatrixCase(
            name="config_stays_available_when_db_is_down",
            method="GET",
            path="/api/v1/config",
            auth_enabled=True,
            db_ok=False,
            pipeline_running=True,
            auth_header="Bearer secret",
            include_clip=False,
            expected_status=200,
        ),
        _MatrixCase(
            name="config_route_is_blocked_in_bootstrap_mode",
            method="GET",
            path="/api/v1/config",
            auth_enabled=False,
            db_ok=False,
            pipeline_running=False,
            auth_header=None,
            include_clip=False,
            expected_status=503,
            bootstrap_mode=True,
            expected_error_code="SETUP_REQUIRED",
        ),
        _MatrixCase(
            name="storage_backends_remain_available_in_bootstrap_mode",
            method="GET",
            path="/api/v1/storage/backends",
            auth_enabled=False,
            db_ok=False,
            pipeline_running=False,
            auth_header=None,
            include_clip=False,
            expected_status=200,
            bootstrap_mode=True,
        ),
        _MatrixCase(
            name="cameras_requires_api_key_when_auth_enabled",
            method="GET",
            path="/api/v1/cameras",
            auth_enabled=True,
            db_ok=True,
            pipeline_running=True,
            auth_header=None,
            include_clip=False,
            expected_status=401,
            expected_error_code="UNAUTHORIZED",
        ),
        _MatrixCase(
            name="cameras_stays_available_when_db_is_down",
            method="GET",
            path="/api/v1/cameras",
            auth_enabled=True,
            db_ok=False,
            pipeline_running=True,
            auth_header="Bearer secret",
            include_clip=False,
            expected_status=200,
        ),
        _MatrixCase(
            name="cameras_route_is_blocked_in_bootstrap_mode",
            method="GET",
            path="/api/v1/cameras",
            auth_enabled=False,
            db_ok=False,
            pipeline_running=False,
            auth_header=None,
            include_clip=False,
            expected_status=503,
            bootstrap_mode=True,
            expected_error_code="SETUP_REQUIRED",
        ),
        _MatrixCase(
            name="runtime_status_requires_api_key_when_auth_enabled",
            method="GET",
            path="/api/v1/runtime/status",
            auth_enabled=True,
            db_ok=True,
            pipeline_running=True,
            auth_header=None,
            include_clip=False,
            expected_status=401,
            expected_error_code="UNAUTHORIZED",
        ),
        _MatrixCase(
            name="runtime_status_does_not_require_db",
            method="GET",
            path="/api/v1/runtime/status",
            auth_enabled=True,
            db_ok=False,
            pipeline_running=True,
            auth_header="Bearer secret",
            include_clip=False,
            expected_status=200,
        ),
        _MatrixCase(
            name="runtime_status_is_blocked_in_bootstrap_mode",
            method="GET",
            path="/api/v1/runtime/status",
            auth_enabled=False,
            db_ok=True,
            pipeline_running=False,
            auth_header=None,
            include_clip=False,
            expected_status=503,
            bootstrap_mode=True,
            expected_error_code="SETUP_REQUIRED",
        ),
        _MatrixCase(
            name="diagnostics_requires_api_key_when_auth_enabled",
            method="GET",
            path="/api/v1/diagnostics",
            auth_enabled=True,
            db_ok=True,
            pipeline_running=True,
            auth_header=None,
            include_clip=False,
            expected_status=401,
            expected_error_code="UNAUTHORIZED",
        ),
        _MatrixCase(
            name="diagnostics_does_not_fail_closed_on_db_outage",
            method="GET",
            path="/api/v1/diagnostics",
            auth_enabled=True,
            db_ok=False,
            pipeline_running=True,
            auth_header="Bearer secret",
            include_clip=False,
            expected_status=200,
        ),
        _MatrixCase(
            name="diagnostics_is_blocked_in_bootstrap_mode",
            method="GET",
            path="/api/v1/diagnostics",
            auth_enabled=False,
            db_ok=True,
            pipeline_running=False,
            auth_header=None,
            include_clip=False,
            expected_status=503,
            bootstrap_mode=True,
            expected_error_code="SETUP_REQUIRED",
        ),
        _MatrixCase(
            name="clips_requires_api_key_when_auth_enabled",
            method="GET",
            path="/api/v1/clips",
            auth_enabled=True,
            db_ok=True,
            pipeline_running=True,
            auth_header=None,
            include_clip=False,
            expected_status=401,
            expected_error_code="UNAUTHORIZED",
        ),
        _MatrixCase(
            name="clips_requires_db_when_repository_unavailable",
            method="GET",
            path="/api/v1/clips",
            auth_enabled=True,
            db_ok=False,
            pipeline_running=True,
            auth_header="Bearer secret",
            include_clip=False,
            expected_status=503,
            expected_error_code="DB_UNAVAILABLE",
        ),
        _MatrixCase(
            name="clips_route_is_blocked_in_bootstrap_mode_before_db_probe",
            method="GET",
            path="/api/v1/clips",
            auth_enabled=False,
            db_ok=False,
            pipeline_running=False,
            auth_header=None,
            include_clip=False,
            expected_status=503,
            bootstrap_mode=True,
            expected_error_code="SETUP_REQUIRED",
        ),
        _MatrixCase(
            name="stats_requires_db_when_repository_unavailable",
            method="GET",
            path="/api/v1/stats",
            auth_enabled=True,
            db_ok=False,
            pipeline_running=True,
            auth_header="Bearer secret",
            include_clip=False,
            expected_status=503,
            expected_error_code="DB_UNAVAILABLE",
        ),
        _MatrixCase(
            name="stats_route_is_blocked_in_bootstrap_mode",
            method="GET",
            path="/api/v1/stats",
            auth_enabled=False,
            db_ok=False,
            pipeline_running=False,
            auth_header=None,
            include_clip=False,
            expected_status=503,
            bootstrap_mode=True,
            expected_error_code="SETUP_REQUIRED",
        ),
        _MatrixCase(
            name="media_route_rejects_missing_media_auth",
            method="GET",
            path="/api/v1/clips/clip-1/media",
            auth_enabled=True,
            db_ok=True,
            pipeline_running=True,
            auth_header=None,
            include_clip=True,
            expected_status=401,
            expected_error_code="MEDIA_TOKEN_REJECTED",
        ),
        _MatrixCase(
            name="media_route_requires_db_before_media_auth",
            method="GET",
            path="/api/v1/clips/clip-1/media",
            auth_enabled=True,
            db_ok=False,
            pipeline_running=True,
            auth_header="Bearer secret",
            include_clip=True,
            expected_status=503,
            expected_error_code="DB_UNAVAILABLE",
        ),
        _MatrixCase(
            name="media_route_allows_api_key_access",
            method="GET",
            path="/api/v1/clips/clip-1/media",
            auth_enabled=True,
            db_ok=True,
            pipeline_running=True,
            auth_header="Bearer secret",
            include_clip=True,
            expected_status=200,
        ),
        _MatrixCase(
            name="media_route_is_blocked_in_bootstrap_mode",
            method="GET",
            path="/api/v1/clips/clip-1/media",
            auth_enabled=False,
            db_ok=True,
            pipeline_running=False,
            auth_header=None,
            include_clip=True,
            expected_status=503,
            bootstrap_mode=True,
            expected_error_code="SETUP_REQUIRED",
        ),
        _MatrixCase(
            name="media_token_mint_requires_api_key",
            method="POST",
            path="/api/v1/clips/clip-1/media-token",
            auth_enabled=True,
            db_ok=True,
            pipeline_running=True,
            auth_header=None,
            include_clip=True,
            expected_status=401,
            expected_error_code="UNAUTHORIZED",
        ),
        _MatrixCase(
            name="media_token_mint_requires_db",
            method="POST",
            path="/api/v1/clips/clip-1/media-token",
            auth_enabled=True,
            db_ok=False,
            pipeline_running=True,
            auth_header="Bearer secret",
            include_clip=True,
            expected_status=503,
            expected_error_code="DB_UNAVAILABLE",
        ),
        _MatrixCase(
            name="media_token_mint_succeeds_with_api_key_and_db",
            method="POST",
            path="/api/v1/clips/clip-1/media-token",
            auth_enabled=True,
            db_ok=True,
            pipeline_running=True,
            auth_header="Bearer secret",
            include_clip=True,
            expected_status=200,
        ),
    ],
    ids=lambda case: case.name,
)
def test_api_bootstrap_dependency_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: _MatrixCase,
) -> None:
    """API groups should enforce consistent auth/DB bootstrap behavior."""
    # Given: A configured app with matrix-controlled auth, DB, and runtime bootstrap state
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    monkeypatch.setattr(setup_service, "_network_probe", lambda: (True, "DNS resolution succeeded"))
    client, repository = _build_client(tmp_path, case)
    headers: dict[str, str] = {}
    if case.auth_header is not None:
        headers["Authorization"] = case.auth_header

    # When: Requesting the endpoint under the selected matrix conditions
    response = _send_request(client, case, headers)

    # Then: Status and canonical error envelope match the dependency policy contract
    assert response.status_code == case.expected_status
    if case.expected_error_code is not None:
        assert response.json()["error_code"] == case.expected_error_code

    # Then: DB-backed failures perform at least one reachability probe
    if case.expected_error_code == "DB_UNAVAILABLE":
        assert repository.ping_calls >= 1


def test_onvif_discover_is_available_in_bootstrap_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ONVIF discover route should stay available in bootstrap mode for onboarding."""
    # Given: A bootstrap-mode app and a deterministic ONVIF discovery stub
    case = _MatrixCase(
        name="onvif_discover_available_in_bootstrap_mode",
        method="POST",
        path="/api/v1/onvif/discover",
        auth_enabled=False,
        db_ok=False,
        pipeline_running=False,
        auth_header=None,
        include_clip=False,
        expected_status=200,
        bootstrap_mode=True,
    )
    monkeypatch.setattr("homesec.api.routes.onvif.discover_cameras", lambda *_args, **_kwargs: [])
    client, _ = _build_client(tmp_path, case)

    # When: Calling ONVIF discover endpoint without API key in bootstrap mode
    response = client.post(case.path, json={})

    # Then: Endpoint responds successfully for setup-camera discovery flow
    assert response.status_code == 200
    assert response.json() == []
