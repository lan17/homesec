"""Tests for preview control-plane API routes."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from homesec.api.preview_tokens import validate_camera_preview_token
from homesec.api.server import create_app
from homesec.models.config import FastAPIServerConfig, PreviewConfig
from homesec.runtime.errors import PreviewCameraNotFoundError, PreviewRuntimeUnavailableError
from homesec.runtime.models import (
    CameraPreviewStartRefusal,
    CameraPreviewStatus,
    CameraPreviewStopResult,
    PreviewRefusalReason,
    PreviewState,
)
from tests.homesec.ui_dist_stub import ensure_stub_ui_dist


class _StubPreviewApp:
    def __init__(
        self,
        *,
        status: CameraPreviewStatus | None = None,
        ensure_result: CameraPreviewStatus | CameraPreviewStartRefusal | None = None,
        stop_result: CameraPreviewStopResult | None = None,
        server_config: FastAPIServerConfig | None = None,
        preview_config: PreviewConfig | None = None,
        bootstrap_mode: bool = False,
        status_error: Exception | None = None,
        ensure_error: Exception | None = None,
        stop_error: Exception | None = None,
    ) -> None:
        resolved_server = ensure_stub_ui_dist(server_config or FastAPIServerConfig())
        resolved_preview = preview_config or PreviewConfig(enabled=True)
        self._config = SimpleNamespace(server=resolved_server, preview=resolved_preview, cameras=[])
        self._bootstrap_mode = bootstrap_mode
        self._status = status or CameraPreviewStatus(
            camera_name="front",
            enabled=True,
            state=PreviewState.READY,
            viewer_count=2,
        )
        self._ensure_result = ensure_result or self._status
        self._stop_result = stop_result or CameraPreviewStopResult(
            camera_name="front",
            accepted=True,
            state=PreviewState.STOPPING,
        )
        self._status_error = status_error
        self._ensure_error = ensure_error
        self._stop_error = stop_error

    @property
    def config(self):
        return self._config

    @property
    def server_config(self) -> FastAPIServerConfig:
        return self._config.server

    @property
    def bootstrap_mode(self) -> bool:
        return self._bootstrap_mode

    async def get_camera_preview_status(self, camera_name: str) -> CameraPreviewStatus:
        if self._status_error is not None:
            raise self._status_error
        assert camera_name == self._status.camera_name
        return self._status

    async def ensure_camera_preview_active(
        self,
        camera_name: str,
    ) -> CameraPreviewStatus | CameraPreviewStartRefusal:
        if self._ensure_error is not None:
            raise self._ensure_error
        result = self._ensure_result
        if isinstance(result, CameraPreviewStatus):
            assert camera_name == result.camera_name
        else:
            assert camera_name == self._status.camera_name
        return result

    async def force_stop_camera_preview(self, camera_name: str) -> CameraPreviewStopResult:
        if self._stop_error is not None:
            raise self._stop_error
        assert camera_name == self._stop_result.camera_name
        return self._stop_result


def _client(app: _StubPreviewApp) -> TestClient:
    return TestClient(create_app(app))


def test_get_preview_status_returns_runtime_status() -> None:
    """GET /preview/cameras/{camera_name} should return preview status."""
    # Given: A preview-capable app with runtime status
    app = _StubPreviewApp(
        status=CameraPreviewStatus(
            camera_name="front",
            enabled=True,
            state=PreviewState.DEGRADED,
            viewer_count=1,
            degraded_reason="viewer_count_unavailable",
            last_error="segment accounting unavailable",
            idle_shutdown_at=123.0,
        )
    )
    client = _client(app)

    # When: Requesting preview status
    response = client.get("/api/v1/preview/cameras/front")

    # Then: The response mirrors runtime preview status fields
    assert response.status_code == 200
    payload = response.json()
    assert payload["camera_name"] == "front"
    assert payload["enabled"] is True
    assert payload["state"] == "degraded"
    assert payload["viewer_count"] == 1
    assert payload["degraded_reason"] == "viewer_count_unavailable"
    assert payload["last_error"] == "segment accounting unavailable"
    assert payload["idle_shutdown_at"] == 123.0


def test_get_preview_status_returns_404_for_missing_camera() -> None:
    """GET /preview/cameras/{camera_name} should return 404 when camera is unknown."""
    # Given: A preview app that reports unknown camera
    app = _StubPreviewApp(status_error=PreviewCameraNotFoundError("missing"))
    client = _client(app)

    # When: Requesting status for a missing camera
    response = client.get("/api/v1/preview/cameras/missing")

    # Then: The route returns a canonical preview-camera-not-found error
    assert response.status_code == 404
    assert response.json()["error_code"] == "PREVIEW_CAMERA_NOT_FOUND"


def test_post_preview_returns_tokenized_playlist_when_auth_enabled(
    monkeypatch,
) -> None:
    """POST /preview/cameras/{camera_name} should mint a fresh preview token."""
    # Given: Auth enabled with a valid API key and a ready preview status
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    app = _StubPreviewApp(
        status=CameraPreviewStatus(
            camera_name="front door",
            enabled=True,
            state=PreviewState.READY,
            viewer_count=3,
        ),
        ensure_result=CameraPreviewStatus(
            camera_name="front door",
            enabled=True,
            state=PreviewState.READY,
            viewer_count=3,
        ),
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY"),
        preview_config=PreviewConfig(enabled=True, token_ttl_s=45, idle_timeout_s=30.0),
    )
    client = _client(app)

    # When: Ensuring preview is active
    response = client.post(
        "/api/v1/preview/cameras/front door",
        headers={"Authorization": "Bearer secret"},
    )

    # Then: The route returns a token, expiry, and tokenized playlist URL
    assert response.status_code == 200
    payload = response.json()
    assert payload["camera_name"] == "front door"
    assert payload["state"] == "ready"
    assert payload["viewer_count"] == 3
    assert payload["idle_timeout_s"] == 30.0
    assert payload["token"]
    assert payload["token_expires_at"] is not None

    parsed = urlparse(payload["playlist_url"])
    query = parse_qs(parsed.query)
    assert parsed.path == "/api/v1/preview/cameras/front%20door/playlist.m3u8"
    assert query["token"] == [payload["token"]]

    validate_camera_preview_token(
        api_key="secret",
        token=payload["token"],
        camera_name="front door",
        now=datetime.now(UTC),
    )


def test_post_preview_returns_direct_playlist_when_auth_disabled() -> None:
    """POST /preview/cameras/{camera_name} should skip token minting when auth is disabled."""
    # Given: Auth disabled and a ready preview
    app = _StubPreviewApp(
        status=CameraPreviewStatus(
            camera_name="front",
            enabled=True,
            state=PreviewState.READY,
            viewer_count=1,
        ),
        ensure_result=CameraPreviewStatus(
            camera_name="front",
            enabled=True,
            state=PreviewState.READY,
            viewer_count=1,
        ),
        preview_config=PreviewConfig(enabled=True, idle_timeout_s=12.5),
    )
    client = _client(app)

    # When: Ensuring preview is active
    response = client.post("/api/v1/preview/cameras/front")

    # Then: The route returns a direct playlist URL without token metadata
    assert response.status_code == 200
    payload = response.json()
    assert payload["token"] is None
    assert payload["token_expires_at"] is None
    assert payload["playlist_url"] == "/api/v1/preview/cameras/front/playlist.m3u8"
    assert payload["idle_timeout_s"] == 12.5


def test_post_preview_returns_warning_when_degraded() -> None:
    """POST /preview/cameras/{camera_name} should surface degraded warnings."""
    # Given: A degraded preview that remains attachable
    app = _StubPreviewApp(
        status=CameraPreviewStatus(
            camera_name="front",
            enabled=True,
            state=PreviewState.DEGRADED,
            viewer_count=None,
            degraded_reason="viewer_count_unavailable",
            last_error="viewer accounting offline",
        ),
        ensure_result=CameraPreviewStatus(
            camera_name="front",
            enabled=True,
            state=PreviewState.DEGRADED,
            viewer_count=None,
            degraded_reason="viewer_count_unavailable",
            last_error="viewer accounting offline",
        ),
    )
    client = _client(app)

    # When: Ensuring preview is active while degraded
    response = client.post("/api/v1/preview/cameras/front")

    # Then: The response includes a warning for UI display
    assert response.status_code == 200
    assert response.json()["warning"] == "viewer_count_unavailable"


def test_post_preview_refusals_map_to_stable_conflicts() -> None:
    """POST /preview/cameras/{camera_name} should return stable refusal codes."""
    # Given: A set of runtime refusal outcomes
    cases = [
        (
            PreviewRefusalReason.RECORDING_PRIORITY,
            "PREVIEW_RECORDING_PRIORITY",
            "recording_priority",
        ),
        (
            PreviewRefusalReason.SESSION_BUDGET_EXHAUSTED,
            "PREVIEW_SESSION_BUDGET_EXHAUSTED",
            "session_budget_exhausted",
        ),
        (
            PreviewRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE,
            "PREVIEW_TEMPORARILY_UNAVAILABLE",
            "preview_temporarily_unavailable",
        ),
    ]

    for reason, error_code, reason_value in cases:
        app = _StubPreviewApp(
            status=CameraPreviewStatus(
                camera_name="front",
                enabled=True,
                state=PreviewState.IDLE,
            ),
            ensure_result=CameraPreviewStartRefusal(
                camera_name="front",
                reason=reason,
                message="Preview cannot start",
            ),
        )
        client = _client(app)

        # When: Ensuring preview is active but runtime refuses it
        response = client.post("/api/v1/preview/cameras/front")

        # Then: The refusal becomes a 409 with stable code and reason
        assert response.status_code == 409
        payload = response.json()
        assert payload["error_code"] == error_code
        assert payload["reason"] == reason_value


def test_delete_preview_returns_202_acknowledgement() -> None:
    """DELETE /preview/cameras/{camera_name} should return async stop acknowledgement."""
    # Given: A preview-capable app that can accept force-stop
    app = _StubPreviewApp(
        stop_result=CameraPreviewStopResult(
            camera_name="front",
            accepted=True,
            state=PreviewState.STOPPING,
        )
    )
    client = _client(app)

    # When: Force-stopping preview
    response = client.delete("/api/v1/preview/cameras/front")

    # Then: The route returns a minimal accepted payload
    assert response.status_code == 202
    assert response.json() == {"accepted": True, "state": "stopping"}


def test_delete_preview_returns_503_when_runtime_unavailable() -> None:
    """DELETE /preview/cameras/{camera_name} should return 503 when runtime is unavailable."""
    # Given: A preview app whose runtime cannot accept stop commands
    app = _StubPreviewApp(stop_error=PreviewRuntimeUnavailableError("worker unavailable"))
    client = _client(app)

    # When: Force-stopping preview
    response = client.delete("/api/v1/preview/cameras/front")

    # Then: The route returns a canonical runtime-unavailable error
    assert response.status_code == 503
    assert response.json()["error_code"] == "PREVIEW_RUNTIME_UNAVAILABLE"
