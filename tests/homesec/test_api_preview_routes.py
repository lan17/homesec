"""Tests for preview control-plane and playback API routes."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from homesec.api.errors import APIError
from homesec.api.preview_tokens import validate_camera_preview_token
from homesec.api.routes import preview as preview_routes
from homesec.api.server import create_app
from homesec.models.config import FastAPIServerConfig, HLSPreviewConfig, PreviewConfig
from homesec.preview_paths import preview_camera_dir, preview_segment_path
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
        camera_names: list[str] | None = None,
        camera_configs: list[SimpleNamespace] | None = None,
        bootstrap_mode: bool = False,
        pipeline_running: bool = True,
        status_error: Exception | None = None,
        ensure_error: Exception | None = None,
        stop_error: Exception | None = None,
    ) -> None:
        resolved_server = ensure_stub_ui_dist(server_config or FastAPIServerConfig())
        resolved_preview = preview_config or PreviewConfig(enabled=True)
        self._bootstrap_mode = bootstrap_mode
        self._pipeline_running = pipeline_running
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
        self.viewer_activity_calls: list[tuple[str, str | None]] = []
        resolved_cameras = camera_configs
        if resolved_cameras is None:
            resolved_camera_names = camera_names or [self._status.camera_name]
            resolved_cameras = [
                SimpleNamespace(
                    name=name,
                    enabled=True,
                    source=SimpleNamespace(backend="rtsp"),
                )
                for name in resolved_camera_names
            ]
        self._config = SimpleNamespace(
            server=resolved_server,
            preview=resolved_preview,
            cameras=resolved_cameras,
        )

    @property
    def config(self):
        return self._config

    @property
    def server_config(self) -> FastAPIServerConfig:
        return self._config.server

    @property
    def bootstrap_mode(self) -> bool:
        return self._bootstrap_mode

    @property
    def pipeline_running(self) -> bool:
        return self._pipeline_running

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

    async def note_camera_preview_viewer_activity(
        self,
        camera_name: str,
        *,
        viewer_id: str | None = None,
    ) -> None:
        self.viewer_activity_calls.append((camera_name, viewer_id))


def _write_preview_files(tmp_path: Path, camera_name: str) -> None:
    camera_dir = preview_camera_dir(tmp_path, camera_name)
    camera_dir.mkdir(parents=True, exist_ok=True)
    (camera_dir / "playlist.m3u8").write_text(
        "#EXTM3U\n#EXT-X-VERSION:3\n#EXTINF:1.0,\nsegment_000000.ts\n",
        encoding="utf-8",
    )
    (camera_dir / "segment_000000.ts").write_bytes(b"segment-bytes")


def _client(app: _StubPreviewApp) -> TestClient:
    return TestClient(create_app(app))


def test_warning_prefers_last_error_and_generic_fallback() -> None:
    """Preview warnings should fall back from degraded_reason to last_error to a generic message."""
    # Given: Degraded preview states with progressively less detail
    last_error_only = CameraPreviewStatus(
        camera_name="front",
        enabled=True,
        state=PreviewState.DEGRADED,
        degraded_reason=None,
        last_error="ffmpeg stalled",
    )
    generic_only = CameraPreviewStatus(
        camera_name="front",
        enabled=True,
        state=PreviewState.DEGRADED,
        degraded_reason=None,
        last_error=None,
    )

    # When: Rendering warning text for each degraded status
    last_error_warning = preview_routes._warning(last_error_only)
    generic_warning = preview_routes._warning(generic_only)

    # Then: The module falls back to last_error first and a generic message last
    assert last_error_warning == "ffmpeg stalled"
    assert generic_warning == "Preview is degraded"


def test_rewrite_playlist_for_token_returns_original_text_when_token_is_absent() -> None:
    """Playlist rewriting should be a no-op when playback is using header auth or auth is disabled."""
    # Given: A preview playlist with a trailing newline
    playlist_text = "#EXTM3U\n#EXTINF:1.0,\nsegment_000000.ts\n"

    # When: Rewriting it without a preview token
    rewritten = preview_routes._rewrite_playlist_for_token(playlist_text, None)

    # Then: The original playlist text is preserved exactly
    assert rewritten == playlist_text


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


def test_get_preview_status_returns_503_when_runtime_unavailable() -> None:
    """GET /preview/cameras/{camera_name} should surface runtime-unavailable failures."""
    # Given: A preview app whose runtime cannot accept status commands
    app = _StubPreviewApp(status_error=PreviewRuntimeUnavailableError("worker unavailable"))
    client = _client(app)

    # When: Requesting preview status
    response = client.get("/api/v1/preview/cameras/front")

    # Then: The route returns the canonical preview-runtime-unavailable error
    assert response.status_code == 503
    assert response.json()["error_code"] == "PREVIEW_RUNTIME_UNAVAILABLE"


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


def test_preview_tokenized_playlist_url_is_playable_end_to_end(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """POST /preview should return a playlist URL that serves playlist and segment media."""
    # Given: Auth enabled with preview artifacts ready on disk
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    camera_name = "front door"
    _write_preview_files(tmp_path, camera_name)
    app = _StubPreviewApp(
        status=CameraPreviewStatus(
            camera_name=camera_name,
            enabled=True,
            state=PreviewState.READY,
            viewer_count=0,
        ),
        ensure_result=CameraPreviewStatus(
            camera_name=camera_name,
            enabled=True,
            state=PreviewState.READY,
            viewer_count=0,
        ),
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY"),
        preview_config=PreviewConfig(
            enabled=True,
            token_ttl_s=45,
            idle_timeout_s=30.0,
            config=HLSPreviewConfig(storage_dir=tmp_path),
        ),
    )
    client = _client(app)

    # When: Starting preview, fetching the returned playlist, then fetching a segment from it
    create_response = client.post(
        "/api/v1/preview/cameras/front door",
        headers={"Authorization": "Bearer secret"},
    )
    assert create_response.status_code == 200
    create_payload = create_response.json()

    playlist_response = client.get(create_payload["playlist_url"])
    playlist_lines = [line for line in playlist_response.text.splitlines() if line]
    segment_url = next(line for line in playlist_lines if not line.startswith("#"))
    segment_response = client.get(
        f"/api/v1/preview/cameras/front door/{segment_url}",
    )

    # Then: Playback succeeds and the segment request records viewer activity using the preview token
    assert playlist_response.status_code == 200
    assert playlist_response.headers["content-type"].startswith("application/vnd.apple.mpegurl")
    assert "segment_000000.ts?token=" in playlist_response.text
    assert segment_response.status_code == 200
    assert segment_response.content == b"segment-bytes"
    assert app.viewer_activity_calls == [("front door", create_payload["token"])]


@pytest.mark.asyncio
async def test_preview_segment_starts_viewer_activity_without_blocking_response_creation(
    tmp_path: Path,
) -> None:
    """Segment responses should start viewer bookkeeping promptly without blocking."""
    # Given: Active preview media and a viewer-activity callback that blocks until released
    _write_preview_files(tmp_path, "front")
    app = _StubPreviewApp(
        preview_config=PreviewConfig(
            enabled=True,
            config=HLSPreviewConfig(storage_dir=tmp_path),
        )
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def _blocking_note(
        camera_name: str,
        *,
        viewer_id: str | None = None,
    ) -> None:
        started.set()
        await release.wait()
        app.viewer_activity_calls.append((camera_name, viewer_id))

    app.note_camera_preview_viewer_activity = _blocking_note

    # When: Building the segment response directly from the route handler
    response = await preview_routes.get_preview_segment(
        camera_name="front",
        segment_name="segment_000000.ts",
        preview_token="preview-token",
        app=app,
    )

    # Then: The route returns immediately and the bookkeeping task starts without waiting
    assert app.viewer_activity_calls == []
    assert response.background is None
    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert app.viewer_activity_calls == []
    release.set()
    await asyncio.sleep(0)
    assert app.viewer_activity_calls == [("front", "preview-token")]


@pytest.mark.asyncio
async def test_preview_segment_rejects_path_traversal_before_path_construction(
    tmp_path: Path,
) -> None:
    """Preview segment route should reject traversal payloads before building a file path."""
    # Given: Active preview media for a valid camera
    _write_preview_files(tmp_path, "front")
    app = _StubPreviewApp(
        preview_config=PreviewConfig(
            enabled=True,
            config=HLSPreviewConfig(storage_dir=tmp_path),
        )
    )

    # When: Requesting a traversal-like segment name directly through the route handler
    with pytest.raises(APIError) as exc_info:
        await preview_routes.get_preview_segment(
            camera_name="front",
            segment_name="../../etc/passwd",
            preview_token=None,
            app=app,
        )

    # Then: The route rejects it with the canonical not-found error before any path join
    assert exc_info.value.status_code == 404
    assert exc_info.value.error_code == "NOT_FOUND"


def test_preview_segment_path_rejects_invalid_segment_names(tmp_path: Path) -> None:
    """Preview segment path helper should reject traversal-like names."""
    # Given: A preview storage root and an invalid segment name
    storage_dir = tmp_path

    # When: Resolving a segment path for a traversal-like segment name
    with pytest.raises(ValueError, match="Invalid preview segment name"):
        preview_segment_path(storage_dir, "front", "../../etc/passwd")

    # Then: No path is constructed outside the preview directory


def test_preview_playlist_rejects_unknown_camera_before_touching_disk(tmp_path: Path) -> None:
    """Preview playlist route should reject unknown cameras before reading playlist files."""
    # Given: Preview is enabled but the requested camera is not configured
    _write_preview_files(tmp_path, "front")
    app = _StubPreviewApp(
        preview_config=PreviewConfig(
            enabled=True,
            config=HLSPreviewConfig(storage_dir=tmp_path),
        ),
        camera_names=["front"],
    )
    client = _client(app)

    # When: Fetching the playlist for an unknown camera
    response = client.get("/api/v1/preview/cameras/garage/playlist.m3u8")

    # Then: The route returns the canonical camera-not-found error
    assert response.status_code == 404
    assert response.json()["error_code"] == "PREVIEW_CAMERA_NOT_FOUND"


def test_preview_playlist_rejects_invalid_token(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Preview playlist route should reject invalid token query strings."""
    # Given: Auth enabled with preview artifacts present
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    _write_preview_files(tmp_path, "front")
    app = _StubPreviewApp(
        preview_config=PreviewConfig(
            enabled=True,
            config=HLSPreviewConfig(storage_dir=tmp_path),
        ),
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY"),
    )
    client = _client(app)

    # When: Fetching the playlist with an invalid preview token
    response = client.get("/api/v1/preview/cameras/front/playlist.m3u8?token=bad-token")

    # Then: The route rejects the request with the canonical preview token error
    assert response.status_code == 401
    assert response.json()["error_code"] == "PREVIEW_TOKEN_REJECTED"


def test_preview_playlist_returns_conflict_when_media_is_missing(tmp_path: Path) -> None:
    """Preview playlist route should report unavailable media until artifacts exist."""
    # Given: Preview is configured but the live playlist is not present yet
    app = _StubPreviewApp(
        preview_config=PreviewConfig(
            enabled=True,
            config=HLSPreviewConfig(storage_dir=tmp_path),
        )
    )
    client = _client(app)

    # When: Fetching the playlist before preview artifacts are ready
    response = client.get("/api/v1/preview/cameras/front/playlist.m3u8")

    # Then: The route reports preview media unavailable instead of falling through to 404
    assert response.status_code == 409
    assert response.json()["error_code"] == "PREVIEW_MEDIA_UNAVAILABLE"


def test_preview_playlist_returns_conflict_when_playlist_is_empty(tmp_path: Path) -> None:
    """Preview playlist route should reject empty playlist files as unavailable media."""
    # Given: A preview directory with an empty playlist file
    camera_dir = preview_camera_dir(tmp_path, "front")
    camera_dir.mkdir(parents=True, exist_ok=True)
    (camera_dir / "playlist.m3u8").write_text("", encoding="utf-8")
    app = _StubPreviewApp(
        preview_config=PreviewConfig(
            enabled=True,
            config=HLSPreviewConfig(storage_dir=tmp_path),
        )
    )
    client = _client(app)

    # When: Fetching the empty playlist
    response = client.get("/api/v1/preview/cameras/front/playlist.m3u8")

    # Then: The route reports preview media unavailable
    assert response.status_code == 409
    assert response.json()["error_code"] == "PREVIEW_MEDIA_UNAVAILABLE"


def test_preview_playlist_returns_conflict_when_playlist_read_fails(
    monkeypatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Preview playlist route should fail closed when reading playlist text raises."""
    # Given: A preview directory whose playlist read fails unexpectedly
    _write_preview_files(tmp_path, "front")
    app = _StubPreviewApp(
        preview_config=PreviewConfig(
            enabled=True,
            config=HLSPreviewConfig(storage_dir=tmp_path),
        )
    )
    client = _client(app)

    def _raise_read_error(self: Path, encoding: str = "utf-8") -> str:
        _ = encoding
        raise OSError("disk read failed")

    monkeypatch.setattr(Path, "read_text", _raise_read_error)

    # When: Fetching the playlist while the file read fails
    with caplog.at_level(logging.WARNING):
        response = client.get("/api/v1/preview/cameras/front/playlist.m3u8")

    # Then: The route reports preview media unavailable and logs the read failure
    assert response.status_code == 409
    assert response.json()["error_code"] == "PREVIEW_MEDIA_UNAVAILABLE"
    assert "Failed to read preview playlist" in caplog.text


def test_preview_playlist_rejects_stale_files_when_preview_disabled(tmp_path: Path) -> None:
    """Preview playback should stay unavailable when preview is disabled in config."""
    # Given: Stale preview artifacts on disk while preview is disabled
    _write_preview_files(tmp_path, "front")
    app = _StubPreviewApp(
        preview_config=PreviewConfig(
            enabled=False,
            config=HLSPreviewConfig(storage_dir=tmp_path),
        )
    )
    client = _client(app)

    # When: Fetching a preview playlist directly
    response = client.get("/api/v1/preview/cameras/front/playlist.m3u8")

    # Then: The route refuses playback instead of serving stale media
    assert response.status_code == 409
    assert response.json()["error_code"] == "PREVIEW_TEMPORARILY_UNAVAILABLE"


def test_preview_playlist_rejects_stale_files_when_runtime_is_unavailable(tmp_path: Path) -> None:
    """Preview playback should fail closed once the runtime is no longer healthy."""
    # Given: Preview artifacts still exist on disk after the runtime has gone unavailable
    _write_preview_files(tmp_path, "front")
    app = _StubPreviewApp(
        preview_config=PreviewConfig(
            enabled=True,
            config=HLSPreviewConfig(storage_dir=tmp_path),
        ),
        pipeline_running=False,
    )
    client = _client(app)

    # When: Fetching the preview playlist directly
    response = client.get("/api/v1/preview/cameras/front/playlist.m3u8")

    # Then: Playback is rejected with the canonical runtime-unavailable error
    assert response.status_code == 503
    assert response.json()["error_code"] == "PREVIEW_RUNTIME_UNAVAILABLE"


def test_preview_playlist_returns_404_when_runtime_reports_missing_camera(tmp_path: Path) -> None:
    """Preview playback should surface missing-camera runtime errors as 404."""
    # Given: Preview artifacts exist but runtime reports the camera no longer exists
    _write_preview_files(tmp_path, "front")
    app = _StubPreviewApp(
        preview_config=PreviewConfig(
            enabled=True,
            config=HLSPreviewConfig(storage_dir=tmp_path),
        ),
        status_error=PreviewCameraNotFoundError("front"),
    )
    client = _client(app)

    # When: Fetching the preview playlist
    response = client.get("/api/v1/preview/cameras/front/playlist.m3u8")

    # Then: Playback surfaces the canonical preview-camera-not-found error
    assert response.status_code == 404
    assert response.json()["error_code"] == "PREVIEW_CAMERA_NOT_FOUND"


def test_preview_playlist_returns_503_when_runtime_status_lookup_is_unavailable(
    tmp_path: Path,
) -> None:
    """Preview playback should surface runtime status failures as 503."""
    # Given: Preview artifacts exist but runtime status cannot be fetched
    _write_preview_files(tmp_path, "front")
    app = _StubPreviewApp(
        preview_config=PreviewConfig(
            enabled=True,
            config=HLSPreviewConfig(storage_dir=tmp_path),
        ),
        status_error=PreviewRuntimeUnavailableError("worker unavailable"),
    )
    client = _client(app)

    # When: Fetching the preview playlist
    response = client.get("/api/v1/preview/cameras/front/playlist.m3u8")

    # Then: Playback surfaces the canonical preview-runtime-unavailable error
    assert response.status_code == 503
    assert response.json()["error_code"] == "PREVIEW_RUNTIME_UNAVAILABLE"


def test_preview_playback_rejects_stale_files_when_runtime_reports_inactive_preview(
    tmp_path: Path,
) -> None:
    """Preview playback should fail closed when runtime status says the session is inactive."""
    # Given: Stale preview artifacts on disk after the runtime has already marked preview idle
    _write_preview_files(tmp_path, "front")
    app = _StubPreviewApp(
        status=CameraPreviewStatus(
            camera_name="front",
            enabled=True,
            state=PreviewState.IDLE,
            viewer_count=0,
        ),
        preview_config=PreviewConfig(
            enabled=True,
            config=HLSPreviewConfig(storage_dir=tmp_path),
        ),
    )
    client = _client(app)

    # When: Fetching playback media directly from the stale on-disk HLS window
    response = client.get("/api/v1/preview/cameras/front/segment_000000.ts")

    # Then: Playback is rejected instead of serving media from an inactive preview session
    assert response.status_code == 409
    assert response.json()["error_code"] == "PREVIEW_MEDIA_UNAVAILABLE"
    assert app.viewer_activity_calls == []


def test_preview_segment_returns_conflict_when_segment_is_missing(tmp_path: Path) -> None:
    """Preview segment route should report missing live-window segments as unavailable media."""
    # Given: A preview session whose playlist exists but the requested segment has already been removed
    _write_preview_files(tmp_path, "front")
    (preview_camera_dir(tmp_path, "front") / "segment_000000.ts").unlink()
    app = _StubPreviewApp(
        preview_config=PreviewConfig(
            enabled=True,
            config=HLSPreviewConfig(storage_dir=tmp_path),
        )
    )
    client = _client(app)

    # When: Fetching the missing segment
    response = client.get("/api/v1/preview/cameras/front/segment_000000.ts")

    # Then: The route reports preview media unavailable
    assert response.status_code == 409
    assert response.json()["error_code"] == "PREVIEW_MEDIA_UNAVAILABLE"


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


def test_post_preview_returns_warning_from_last_error_when_degraded_reason_is_missing() -> None:
    """POST /preview/cameras/{camera_name} should fall back to last_error in warnings."""
    # Given: A degraded preview with no explicit degraded_reason
    degraded_status = CameraPreviewStatus(
        camera_name="front",
        enabled=True,
        state=PreviewState.DEGRADED,
        viewer_count=None,
        degraded_reason=None,
        last_error="playlist lagging",
    )
    app = _StubPreviewApp(status=degraded_status, ensure_result=degraded_status)
    client = _client(app)

    # When: Ensuring preview is active while degraded
    response = client.post("/api/v1/preview/cameras/front")

    # Then: The response falls back to last_error for the warning text
    assert response.status_code == 200
    assert response.json()["warning"] == "playlist lagging"


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


def test_post_preview_returns_404_when_runtime_reports_missing_camera() -> None:
    """POST /preview/cameras/{camera_name} should map runtime missing-camera errors to 404."""
    # Given: A preview app whose runtime cannot find the requested camera
    app = _StubPreviewApp(ensure_error=PreviewCameraNotFoundError("front"))
    client = _client(app)

    # When: Ensuring preview is active
    response = client.post("/api/v1/preview/cameras/front")

    # Then: The route returns the canonical preview-camera-not-found error
    assert response.status_code == 404
    assert response.json()["error_code"] == "PREVIEW_CAMERA_NOT_FOUND"


def test_post_preview_returns_503_when_runtime_is_unavailable() -> None:
    """POST /preview/cameras/{camera_name} should map runtime outages to 503."""
    # Given: A preview app whose runtime cannot accept preview-start commands
    app = _StubPreviewApp(ensure_error=PreviewRuntimeUnavailableError("worker unavailable"))
    client = _client(app)

    # When: Ensuring preview is active
    response = client.post("/api/v1/preview/cameras/front")

    # Then: The route returns the canonical preview-runtime-unavailable error
    assert response.status_code == 503
    assert response.json()["error_code"] == "PREVIEW_RUNTIME_UNAVAILABLE"


def test_post_preview_returns_500_when_auth_is_enabled_but_api_key_is_missing(
    monkeypatch,
) -> None:
    """POST /preview/cameras/{camera_name} should fail closed when auth is enabled without a key."""
    # Given: Auth is enabled but the configured API key environment variable is unset
    monkeypatch.delenv("HOMESEC_PREVIEW_TEST_MISSING_API_KEY", raising=False)
    app = _StubPreviewApp(
        server_config=FastAPIServerConfig(
            auth_enabled=True,
            api_key_env="HOMESEC_PREVIEW_TEST_MISSING_API_KEY",
        ),
    )
    client = _client(app)

    # When: Ensuring preview is active
    response = client.post("/api/v1/preview/cameras/front")

    # Then: The route reports the canonical API-key-not-configured error
    assert response.status_code == 500
    assert response.json()["error_code"] == "API_KEY_NOT_CONFIGURED"


@pytest.mark.asyncio
async def test_post_preview_direct_call_rejects_missing_api_key_after_runtime_success(
    monkeypatch,
) -> None:
    """Direct preview attach logic should still fail closed when auth is enabled without a key."""
    # Given: Runtime attach succeeds but the configured preview-signing key is unavailable
    monkeypatch.delenv("HOMESEC_PREVIEW_TEST_MISSING_API_KEY", raising=False)
    app = _StubPreviewApp(
        server_config=FastAPIServerConfig(
            auth_enabled=True,
            api_key_env="HOMESEC_PREVIEW_TEST_MISSING_API_KEY",
        ),
    )

    # When: Calling the route handler directly after runtime attach succeeds
    with pytest.raises(APIError) as exc_info:
        await preview_routes.ensure_preview_active(camera_name="front", app=app)

    # Then: The handler rejects the request with the canonical API-key-not-configured error
    assert exc_info.value.status_code == 500
    assert exc_info.value.error_code == "API_KEY_NOT_CONFIGURED"


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


def test_delete_preview_returns_404_when_camera_is_missing() -> None:
    """DELETE /preview/cameras/{camera_name} should return 404 for missing cameras."""
    # Given: A preview app whose runtime cannot find the requested camera
    app = _StubPreviewApp(stop_error=PreviewCameraNotFoundError("front"))
    client = _client(app)

    # When: Force-stopping preview
    response = client.delete("/api/v1/preview/cameras/front")

    # Then: The route returns the canonical preview-camera-not-found error
    assert response.status_code == 404
    assert response.json()["error_code"] == "PREVIEW_CAMERA_NOT_FOUND"


@pytest.mark.asyncio
async def test_note_preview_viewer_activity_best_effort_logs_missing_camera(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Viewer-activity bookkeeping should log and swallow missing-camera races."""
    # Given: A preview app whose viewer-activity callback races with camera removal
    app = _StubPreviewApp()

    async def _raise_missing_camera(
        camera_name: str,
        *,
        viewer_id: str | None = None,
    ) -> None:
        _ = (camera_name, viewer_id)
        raise PreviewCameraNotFoundError("front")

    app.note_camera_preview_viewer_activity = _raise_missing_camera

    # When: Recording viewer activity best-effort
    with caplog.at_level(logging.WARNING):
        await preview_routes._note_preview_viewer_activity_best_effort(
            app,
            "front",
            viewer_id="viewer-1",
        )

    # Then: The helper logs the race and swallows the exception
    assert "Preview viewer activity camera disappeared" in caplog.text


@pytest.mark.asyncio
async def test_note_preview_viewer_activity_best_effort_logs_runtime_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Viewer-activity bookkeeping should log and swallow runtime outages."""
    # Given: A preview app whose viewer-activity callback cannot reach the runtime
    app = _StubPreviewApp()

    async def _raise_runtime_unavailable(
        camera_name: str,
        *,
        viewer_id: str | None = None,
    ) -> None:
        _ = (camera_name, viewer_id)
        raise PreviewRuntimeUnavailableError("worker unavailable")

    app.note_camera_preview_viewer_activity = _raise_runtime_unavailable

    # When: Recording viewer activity best-effort
    with caplog.at_level(logging.WARNING):
        await preview_routes._note_preview_viewer_activity_best_effort(
            app,
            "front",
            viewer_id="viewer-1",
        )

    # Then: The helper logs the outage and swallows the exception
    assert "Preview viewer activity update skipped" in caplog.text


@pytest.mark.asyncio
async def test_preview_segment_maps_defensive_path_validation_failures_to_404(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Preview segment route should preserve a 404 even if path helper validation fails unexpectedly."""
    # Given: A valid-looking segment request whose path helper still rejects the segment name
    _write_preview_files(tmp_path, "front")
    app = _StubPreviewApp(
        preview_config=PreviewConfig(
            enabled=True,
            config=HLSPreviewConfig(storage_dir=tmp_path),
        )
    )

    def _raise_value_error(storage_dir: Path, camera_name: str, segment_name: str) -> Path:
        _ = (storage_dir, camera_name, segment_name)
        raise ValueError("segment path rejected")

    monkeypatch.setattr(preview_routes, "preview_segment_path", _raise_value_error)

    # When: Fetching the preview segment directly from the route handler
    with pytest.raises(APIError) as exc_info:
        await preview_routes.get_preview_segment(
            camera_name="front",
            segment_name="segment_000000.ts",
            preview_token=None,
            app=app,
        )

    # Then: The route preserves the canonical not-found response
    assert exc_info.value.status_code == 404
    assert exc_info.value.error_code == "NOT_FOUND"
