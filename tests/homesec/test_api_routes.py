"""Tests for FastAPI camera and clip routes."""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
import yaml
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from pydantic import BaseModel

from homesec.api.server import create_app
from homesec.config.manager import ConfigManager
from homesec.models.alert import AlertDecision
from homesec.models.clip import ClipListCursor, ClipListPage, ClipStateData
from homesec.models.config import CameraConfig, CameraSourceConfig, FastAPIServerConfig
from homesec.models.enums import ClipStatus, RiskLevel
from homesec.models.filter import FilterResult
from homesec.models.vlm import AnalysisResult


class _StubRepository:
    def __init__(
        self,
        clips: list[ClipStateData] | None = None,
        ok: bool = True,
        clips_count: int | None = None,
        alerts_count: int | None = None,
        list_clips_handler: Callable[..., ClipListPage] | None = None,
    ) -> None:
        self._clips = clips or []
        self._ok = ok
        self._clips_count = clips_count
        self._alerts_count = alerts_count
        self._list_clips_handler = list_clips_handler
        self.deleted_clip_ids: list[str] = []
        self.list_clips_calls: list[dict[str, object]] = []
        self.ping_calls = 0

    async def ping(self) -> bool:
        self.ping_calls += 1
        return self._ok

    async def list_clips(
        self,
        *,
        camera: str | None = None,
        status: ClipStatus | None = None,
        alerted: bool | None = None,
        detected: bool | None = None,
        risk_level: str | None = None,
        activity_type: str | None = None,
        since: dt.datetime | None = None,
        until: dt.datetime | None = None,
        cursor: ClipListCursor | None = None,
        limit: int = 50,
    ) -> ClipListPage:
        call = {
            "camera": camera,
            "status": status,
            "alerted": alerted,
            "detected": detected,
            "risk_level": risk_level,
            "activity_type": activity_type,
            "since": since,
            "until": until,
            "cursor": cursor,
            "limit": limit,
        }
        self.list_clips_calls.append(call)

        if self._list_clips_handler is not None:
            return self._list_clips_handler(**call)

        fetch_limit = max(1, int(limit))
        has_more = len(self._clips) > fetch_limit
        visible = self._clips[:fetch_limit]

        next_cursor = None
        if has_more and visible:
            last = visible[-1]
            if last.created_at is not None and last.clip_id is not None:
                next_cursor = ClipListCursor(created_at=last.created_at, clip_id=last.clip_id)

        return ClipListPage(clips=visible, next_cursor=next_cursor, has_more=has_more)

    async def get_clip(self, clip_id: str) -> ClipStateData | None:
        for clip in self._clips:
            if clip.clip_id == clip_id:
                return clip
        return None

    async def delete_clip(self, clip_id: str) -> ClipStateData:
        self.deleted_clip_ids.append(clip_id)
        clip = await self.get_clip(clip_id)
        if clip is None:
            raise ValueError(f"Clip not found: {clip_id}")
        return clip

    async def count_clips_since(self, since: dt.datetime) -> int:
        _ = since
        if self._clips_count is not None:
            return self._clips_count
        return len(self._clips)

    async def count_alerts_since(self, since: dt.datetime) -> int:
        _ = since
        if self._alerts_count is not None:
            return self._alerts_count
        return 0


class _StubStorage:
    def __init__(
        self,
        *,
        ok: bool = True,
        fail_delete: bool = False,
        fail_get: bool = False,
        media_bytes: bytes = b"",
    ) -> None:
        self._ok = ok
        self._fail_delete = fail_delete
        self._fail_get = fail_get
        self._media_bytes = media_bytes
        self.delete_calls: list[str] = []

    async def ping(self) -> bool:
        return self._ok

    async def get(self, storage_uri: str, local_path: Path) -> None:
        _ = storage_uri
        if self._fail_get:
            raise RuntimeError("get failed")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self._media_bytes)

    async def delete(self, storage_uri: str) -> None:
        self.delete_calls.append(storage_uri)
        _ = storage_uri
        if self._fail_delete:
            raise RuntimeError("delete failed")
        return None


class _StubSource:
    def __init__(self, healthy: bool = True, heartbeat: float = 0.0) -> None:
        self._healthy = healthy
        self._heartbeat = heartbeat

    def is_healthy(self) -> bool:
        return self._healthy

    def last_heartbeat(self) -> float:
        return self._heartbeat


class _StubApp:
    def __init__(
        self,
        *,
        config_manager: ConfigManager,
        repository: _StubRepository,
        storage: _StubStorage,
        sources_by_name: dict[str, _StubSource] | None = None,
        server_config: FastAPIServerConfig | None = None,
        pipeline_running: bool = True,
    ) -> None:
        self.config_manager = config_manager
        self.repository = repository
        self.storage = storage
        self._sources_by_name = sources_by_name or {}
        self.sources = list(self._sources_by_name.values())
        self._config = SimpleNamespace(
            server=server_config or FastAPIServerConfig(),
            cameras=[
                CameraConfig(
                    name=name,
                    enabled=True,
                    source=CameraSourceConfig(
                        backend="local_folder",
                        config={"watch_dir": "/tmp"},
                    ),
                )
                for name in self._sources_by_name
            ],
        )
        self._pipeline_running = pipeline_running
        self.uptime_seconds = 0.0

    @property
    def config(self):  # type: ignore[override]
        return self._config

    @property
    def pipeline_running(self) -> bool:
        return self._pipeline_running

    def get_source(self, camera_name: str) -> _StubSource | None:
        return self._sources_by_name.get(camera_name)


def _write_config(tmp_path, cameras: list[dict]) -> ConfigManager:
    payload = {
        "version": 1,
        "cameras": cameras,
        "storage": {"backend": "dropbox", "config": {"root": "/homecam"}},
        "state_store": {"dsn": "postgresql://user:pass@localhost/db"},
        "notifiers": [{"backend": "mqtt", "config": {"host": "localhost"}}],
        "filter": {"backend": "yolo", "config": {}},
        "vlm": {
            "backend": "openai",
            "config": {"api_key_env": "OPENAI_API_KEY", "model": "gpt-4o"},
        },
        "alert_policy": {"backend": "default", "config": {}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return ConfigManager(path)


def _write_ui_dist(tmp_path: Path) -> Path:
    dist_dir = tmp_path / "ui-dist"
    assets_dir = dist_dir / "assets"
    assets_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("<!doctype html><html><body>HomeSec UI</body></html>")
    (dist_dir / "favicon.ico").write_bytes(b"ico")
    (assets_dir / "app.js").write_text("console.log('homesec ui');")
    (dist_dir / "secrets.map").write_text("{}")
    return dist_dir


def _client(app: _StubApp) -> TestClient:
    return TestClient(create_app(app))


def test_create_camera(tmp_path) -> None:
    """POST /cameras should create a camera."""
    # Given a config with no cameras
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())
    client = _client(app)

    # When creating a camera
    response = client.post(
        "/api/v1/cameras",
        json={
            "name": "front",
            "enabled": True,
            "source_backend": "local_folder",
            "source_config": {"watch_dir": "/tmp"},
        },
    )

    # Then it is created
    assert response.status_code == 201
    payload = response.json()
    assert payload["restart_required"] is True
    assert payload["camera"]["name"] == "front"


def test_create_camera_duplicate_returns_409(tmp_path) -> None:
    """POST /cameras should return 409 for duplicate names."""
    # Given a config with an existing camera
    manager = _write_config(
        tmp_path,
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp"}},
            }
        ],
    )
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())
    client = _client(app)

    # When creating a camera with the same name
    response = client.post(
        "/api/v1/cameras",
        json={
            "name": "front",
            "enabled": True,
            "source_backend": "local_folder",
            "source_config": {"watch_dir": "/tmp"},
        },
    )

    # Then it returns 409 conflict with a canonical error code
    assert response.status_code == 409
    payload = response.json()
    assert payload["error_code"] == "CAMERA_ALREADY_EXISTS"


def test_get_camera(tmp_path) -> None:
    """GET /cameras/{name} should return a camera."""
    # Given a config with one camera
    manager = _write_config(
        tmp_path,
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp"}},
            }
        ],
    )
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())
    client = _client(app)

    # When requesting the camera
    response = client.get("/api/v1/cameras/front")

    # Then it returns the camera
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "front"
    assert payload["enabled"] is True


def test_get_camera_missing_returns_404(tmp_path) -> None:
    """GET /cameras/{name} should return 404 when missing."""
    # Given a config with no cameras
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())
    client = _client(app)

    # When requesting a missing camera
    response = client.get("/api/v1/cameras/missing")

    # Then it returns 404 with canonical error code
    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "CAMERA_NOT_FOUND"


def test_list_cameras_includes_health_fields(tmp_path) -> None:
    """GET /cameras should include health fields."""
    # Given a config with one camera and an unhealthy source
    manager = _write_config(
        tmp_path,
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp"}},
            }
        ],
    )
    sources = {"front": _StubSource(healthy=False, heartbeat=12.3)}
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(),
        storage=_StubStorage(),
        sources_by_name=sources,
    )
    client = _client(app)

    # When listing cameras
    response = client.get("/api/v1/cameras")

    # Then it includes health data
    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["healthy"] is False
    assert payload[0]["last_heartbeat"] == 12.3


def test_list_cameras_serializes_model_config(tmp_path) -> None:
    """GET /cameras should serialize BaseModel configs."""
    # Given a config with a BaseModel-backed source config
    manager = _write_config(
        tmp_path,
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp"}},
            }
        ],
    )
    config = manager.get_config()

    class _ModelConfig(BaseModel):
        watch_dir: str

    config.cameras[0].source.config = _ModelConfig(watch_dir="/tmp")

    class _StaticConfigManager:
        def __init__(self, cfg) -> None:
            self._cfg = cfg

        def get_config(self):
            return self._cfg

    sources = {"front": _StubSource(healthy=True)}
    app = _StubApp(
        config_manager=_StaticConfigManager(config),
        repository=_StubRepository(),
        storage=_StubStorage(),
        sources_by_name=sources,
    )
    client = _client(app)

    # When listing cameras
    response = client.get("/api/v1/cameras")

    # Then the config is serialized as a dict
    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["source_config"]["watch_dir"] == "/tmp"


def test_list_cameras_redacts_sensitive_source_config(tmp_path) -> None:
    """GET /cameras should redact sensitive source_config fields."""
    # Given a camera config loaded from a static manager with direct secret values
    manager = _write_config(
        tmp_path,
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {
                    "backend": "local_folder",
                    "config": {"watch_dir": "/tmp"},
                },
            }
        ],
    )
    config = manager.get_config()
    config.cameras[0].source.config = {
        "rtsp_url": "rtsp://user:pass@10.0.0.5:554/stream",
        "rtsp_url_env": "FRONT_RTSP_URL",
        "password": "top-secret",
        "password_env": "FRONT_PASSWORD",
        "private_key": "super-private",
    }

    class _StaticConfigManager:
        def __init__(self, cfg) -> None:
            self._cfg = cfg

        def get_config(self):
            return self._cfg

    app = _StubApp(
        config_manager=_StaticConfigManager(config),
        repository=_StubRepository(),
        storage=_StubStorage(),
    )
    client = _client(app)

    # When listing cameras
    response = client.get("/api/v1/cameras")

    # Then sensitive fields are redacted and *_env values are preserved
    assert response.status_code == 200
    payload = response.json()
    source_config = payload[0]["source_config"]
    assert source_config["rtsp_url"] == "rtsp://***redacted***@10.0.0.5:554/stream"
    assert source_config["rtsp_url_env"] == "FRONT_RTSP_URL"
    assert source_config["password"] == "***redacted***"
    assert source_config["password_env"] == "FRONT_PASSWORD"
    assert source_config["private_key"] == "***redacted***"


def test_delete_camera(tmp_path) -> None:
    """DELETE /cameras/{name} should remove a camera."""
    # Given a config with one camera
    manager = _write_config(
        tmp_path,
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp"}},
            }
        ],
    )
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())
    client = _client(app)

    # When deleting the camera
    response = client.delete("/api/v1/cameras/front")

    # Then it is removed
    assert response.status_code == 200
    payload = response.json()
    assert payload["restart_required"] is True


def test_delete_camera_missing_returns_404(tmp_path) -> None:
    """DELETE /cameras/{name} should return 404 when missing."""
    # Given a config with no cameras
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())
    client = _client(app)

    # When deleting a missing camera
    response = client.delete("/api/v1/cameras/missing")

    # Then it returns 404 with canonical error code
    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "CAMERA_NOT_FOUND"


def test_update_camera(tmp_path) -> None:
    """PUT /cameras/{name} should update camera fields."""
    # Given a config with one camera
    manager = _write_config(
        tmp_path,
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp"}},
            }
        ],
    )
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())
    client = _client(app)

    # When updating the camera
    response = client.put(
        "/api/v1/cameras/front",
        json={"enabled": False, "source_config": {"watch_dir": "/new"}},
    )

    # Then it returns updated fields
    assert response.status_code == 200
    payload = response.json()
    assert payload["camera"]["enabled"] is False
    assert payload["camera"]["source_config"]["watch_dir"] == "/new"


def test_update_camera_invalid_config_returns_400(tmp_path) -> None:
    """PUT /cameras/{name} should return 400 for invalid config."""
    # Given a config with one camera
    manager = _write_config(
        tmp_path,
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp"}},
            }
        ],
    )
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())
    client = _client(app)

    # When updating with an invalid source config
    response = client.put(
        "/api/v1/cameras/front",
        json={"source_config": {"poll_interval": -1.0}},
    )

    # Then it returns 400 with canonical error code
    assert response.status_code == 400
    payload = response.json()
    assert payload["error_code"] == "CAMERA_CONFIG_INVALID"


def test_update_camera_missing_returns_404(tmp_path) -> None:
    """PUT /cameras/{name} should return 404 when missing."""
    # Given a config with no cameras
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())
    client = _client(app)

    # When updating a missing camera
    response = client.put("/api/v1/cameras/missing", json={"enabled": False})

    # Then it returns 404 with canonical error code
    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "CAMERA_NOT_FOUND"


def test_update_camera_returns_404_when_camera_removed_after_update(tmp_path, monkeypatch) -> None:
    """PUT /cameras/{name} should return 404 when camera disappears after update."""
    # Given a config manager whose update succeeds but subsequent read omits the camera
    manager = _write_config(
        tmp_path,
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp"}},
            }
        ],
    )
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())
    client = _client(app)

    async def _update_camera(
        camera_name: str,
        enabled: bool | None = None,
        source_config: dict[str, object] | None = None,
    ) -> SimpleNamespace:
        _ = (camera_name, enabled, source_config)
        return SimpleNamespace(restart_required=True)

    monkeypatch.setattr(app.config_manager, "update_camera", _update_camera)
    monkeypatch.setattr(app.config_manager, "get_config", lambda: SimpleNamespace(cameras=[]))

    # When updating an existing camera
    response = client.put("/api/v1/cameras/front", json={"enabled": False})

    # Then route returns canonical not-found because camera vanished post-update
    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "CAMERA_NOT_FOUND"


def test_get_config_returns_full_config(tmp_path) -> None:
    """GET /config should return the full configuration."""
    # Given a config with one camera
    manager = _write_config(
        tmp_path,
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp"}},
            }
        ],
    )
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())
    client = _client(app)

    # When requesting config
    response = client.get("/api/v1/config")

    # Then it returns the config payload
    assert response.status_code == 200
    payload = response.json()
    assert payload["config"]["version"] == 1
    assert payload["config"]["cameras"][0]["name"] == "front"
    assert payload["config"]["server"]["enabled"] is True


def test_get_config_redacts_sensitive_fields(tmp_path) -> None:
    """GET /config should redact direct secret values while preserving *_env references."""
    # Given a config containing direct secrets and env var references
    payload = {
        "version": 1,
        "cameras": [
            {
                "name": "front",
                "enabled": True,
                "source": {
                    "backend": "rtsp",
                    "config": {
                        "rtsp_url": "rtsp://user:pass@10.0.0.5:554/stream",
                        "rtsp_url_env": "FRONT_RTSP_URL",
                    },
                },
            }
        ],
        "storage": {"backend": "dropbox", "config": {"root": "/homecam"}},
        "state_store": {"dsn": "postgresql://user:password@localhost/homesec"},
        "notifiers": [
            {
                "backend": "mqtt",
                "config": {
                    "host": "localhost",
                    "auth": {
                        "password_env": "MQTT_PASSWORD",
                        "passphrase": "mqtt-passphrase",
                        "bearer_token": "mqtt-bearer-token",
                    },
                    "connection_string": "postgresql://user:pass@localhost:5432/mqtt",
                },
            }
        ],
        "filter": {"backend": "yolo", "config": {}},
        "vlm": {
            "backend": "openai",
            "config": {"api_key_env": "OPENAI_API_KEY", "model": "gpt-4o"},
        },
        "alert_policy": {"backend": "default", "config": {}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    manager = ConfigManager(path)
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())
    client = _client(app)

    # When requesting config
    response = client.get("/api/v1/config")

    # Then direct secrets are redacted and *_env values are preserved
    assert response.status_code == 200
    config = response.json()["config"]
    assert config["state_store"]["dsn"] == "***redacted***"
    assert (
        config["cameras"][0]["source"]["config"]["rtsp_url"]
        == "rtsp://***redacted***@10.0.0.5:554/stream"
    )
    assert config["cameras"][0]["source"]["config"]["rtsp_url_env"] == "FRONT_RTSP_URL"
    assert config["vlm"]["config"]["api_key_env"] == "OPENAI_API_KEY"
    assert config["notifiers"][0]["config"]["auth"]["passphrase"] == "***redacted***"
    assert config["notifiers"][0]["config"]["auth"]["bearer_token"] == "***redacted***"
    assert config["notifiers"][0]["config"]["connection_string"] == "***redacted***"


def test_get_config_returns_empty_mapping_when_redaction_result_is_not_mapping(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /config should fallback to empty object when redaction returns non-mapping."""
    # Given a valid config and a redaction helper patched to return a list
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())
    client = _client(app)
    monkeypatch.setattr("homesec.api.routes.config._redact_config", lambda value, key=None: [])

    # When requesting config
    response = client.get("/api/v1/config")

    # Then route returns a safe empty config mapping
    assert response.status_code == 200
    assert response.json() == {"config": {}}


def test_cors_disables_credentials_for_wildcard_origins(tmp_path) -> None:
    """CORS should disable credentials when wildcard origins are configured."""
    # Given a server config with wildcard origin
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())

    # When creating FastAPI app
    fastapi_app = create_app(app)

    # Then CORS credentials are disabled to satisfy browser CORS rules
    cors = next(m for m in fastapi_app.user_middleware if m.cls is CORSMiddleware)
    assert cors.kwargs["allow_credentials"] is False


def test_cors_allows_credentials_for_explicit_origins(tmp_path) -> None:
    """CORS should allow credentials when origins are explicit."""
    # Given a server config with explicit origins
    server_config = FastAPIServerConfig(cors_origins=["https://example.com"])
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(),
        storage=_StubStorage(),
        server_config=server_config,
    )

    # When creating FastAPI app
    fastapi_app = create_app(app)

    # Then CORS credentials are enabled
    cors = next(m for m in fastapi_app.user_middleware if m.cls is CORSMiddleware)
    assert cors.kwargs["allow_credentials"] is True


def test_ui_serving_serves_spa_shell_without_shadowing_api(tmp_path) -> None:
    """UI serving should return SPA shell while preserving API and docs routes."""
    # Given a built UI dist directory and UI serving enabled
    dist_dir = _write_ui_dist(tmp_path)
    server_config = FastAPIServerConfig(serve_ui=True, ui_dist_dir=str(dist_dir))
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(),
        storage=_StubStorage(),
        server_config=server_config,
    )
    client = _client(app)

    # When requesting index, a deep SPA path, assets, and API/docs paths
    index_response = client.get("/")
    deep_link_response = client.get("/clips/clip-1")
    asset_response = client.get("/assets/app.js")
    favicon_response = client.get("/favicon.ico")
    map_response = client.get("/secrets.map")
    health_response = client.get("/api/v1/health")
    missing_api_response = client.get("/api/v1/unknown")
    docs_response = client.get("/docs")

    # Then SPA paths serve UI content while API/docs behavior remains intact
    assert index_response.status_code == 200
    assert "HomeSec UI" in index_response.text
    assert deep_link_response.status_code == 200
    assert "HomeSec UI" in deep_link_response.text
    assert asset_response.status_code == 200
    assert "homesec ui" in asset_response.text
    assert favicon_response.status_code == 200
    assert favicon_response.content == b"ico"
    assert map_response.status_code == 404
    assert health_response.status_code == 200
    assert health_response.headers["content-type"].startswith("application/json")
    assert missing_api_response.status_code == 404
    assert missing_api_response.json() == {"detail": "Not Found", "error_code": "NOT_FOUND"}
    assert docs_response.status_code == 200
    assert "Swagger UI" in docs_response.text


def test_ui_serving_skips_routes_when_dist_missing(tmp_path, caplog) -> None:
    """UI serving should no-op with warning when configured dist is missing."""
    # Given UI serving enabled with a missing dist directory
    missing_dist = tmp_path / "missing-dist"
    server_config = FastAPIServerConfig(serve_ui=True, ui_dist_dir=str(missing_dist))
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(),
        storage=_StubStorage(),
        server_config=server_config,
    )

    # When creating app and requesting root
    with caplog.at_level("WARNING"):
        client = _client(app)
    root_response = client.get("/")
    health_response = client.get("/api/v1/health")

    # Then root remains unresolved and a warning is emitted without breaking API routes
    assert root_response.status_code == 404
    assert health_response.status_code == 200
    assert any(
        "UI serving enabled but index file not found" in rec.message for rec in caplog.records
    )


def test_list_clips_cursor_pagination(tmp_path) -> None:
    """GET /clips should paginate with cursor keyset semantics."""
    # Given repository pages keyed by decoded cursor
    now = dt.datetime(2026, 2, 14, 2, 0, tzinfo=dt.timezone.utc)
    clip_a = ClipStateData(
        clip_id="clip-a",
        camera_name="front",
        status=ClipStatus.UPLOADED,
        local_path="/tmp/a.mp4",
        created_at=now,
    )
    clip_b = ClipStateData(
        clip_id="clip-b",
        camera_name="front",
        status=ClipStatus.UPLOADED,
        local_path="/tmp/b.mp4",
        created_at=now,
    )
    clip_c = ClipStateData(
        clip_id="clip-c",
        camera_name="front",
        status=ClipStatus.UPLOADED,
        local_path="/tmp/c.mp4",
        created_at=now - dt.timedelta(seconds=1),
    )
    page_two_cursor = ClipListCursor(created_at=now, clip_id="clip-a")
    first_page = ClipListPage(
        clips=[clip_b, clip_a],
        next_cursor=page_two_cursor,
        has_more=True,
    )
    second_page = ClipListPage(
        clips=[clip_c],
        next_cursor=None,
        has_more=False,
    )

    def list_clips_handler(**kwargs: object) -> ClipListPage:
        cursor = kwargs["cursor"]
        if cursor is None:
            return first_page
        if cursor == page_two_cursor:
            return second_page
        return ClipListPage(clips=[], next_cursor=None, has_more=False)

    manager = _write_config(tmp_path, cameras=[])
    repository = _StubRepository(list_clips_handler=list_clips_handler)
    app = _StubApp(
        config_manager=manager,
        repository=repository,
        storage=_StubStorage(),
    )
    client = _client(app)

    # When requesting the first page
    first_response = client.get("/api/v1/clips?limit=2")

    # Then first page returns two clips and a cursor
    assert first_response.status_code == 200
    first_payload = first_response.json()
    assert first_payload["limit"] == 2
    assert [clip["id"] for clip in first_payload["clips"]] == ["clip-b", "clip-a"]
    assert first_payload["has_more"] is True
    assert isinstance(first_payload["next_cursor"], str)

    # When requesting the next page with the returned cursor
    second_response = client.get(f"/api/v1/clips?limit=2&cursor={first_payload['next_cursor']}")

    # Then second page returns remaining clips and no next cursor
    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert [clip["id"] for clip in second_payload["clips"]] == ["clip-c"]
    assert second_payload["has_more"] is False
    assert second_payload["next_cursor"] is None
    assert repository.list_clips_calls[0]["cursor"] is None
    assert repository.list_clips_calls[1]["cursor"] == page_two_cursor
    assert repository.list_clips_calls[0]["limit"] == 2
    assert repository.list_clips_calls[1]["limit"] == 2


def test_list_clips_invalid_cursor_returns_canonical_error(tmp_path) -> None:
    """GET /clips should return canonical 400 for invalid cursor tokens."""
    # Given a healthy app
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(),
        storage=_StubStorage(),
    )
    client = _client(app)

    # When requesting clips with an invalid cursor token
    response = client.get("/api/v1/clips?cursor=not-a-valid-token")

    # Then it returns canonical bad-request error
    assert response.status_code == 400
    payload = response.json()
    assert payload["error_code"] == "CLIPS_CURSOR_INVALID"


def test_list_clips_rejects_naive_datetime_filter(tmp_path) -> None:
    """GET /clips should reject datetime filters without timezone."""
    # Given a healthy app
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(),
        storage=_StubStorage(),
    )
    client = _client(app)

    # When requesting clips with a naive timestamp
    response = client.get("/api/v1/clips?since=2026-02-14T02:00:00")

    # Then it returns canonical bad-request error
    assert response.status_code == 400
    payload = response.json()
    assert payload["error_code"] == "CLIPS_TIMESTAMP_TZ_REQUIRED"


def test_list_clips_rejects_inverted_time_window(tmp_path) -> None:
    """GET /clips should reject when since is greater than until."""
    # Given a healthy app
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(),
        storage=_StubStorage(),
    )
    client = _client(app)

    # When requesting clips with inverted time window
    response = client.get("/api/v1/clips?since=2026-02-14T03:00:00Z&until=2026-02-14T02:00:00Z")

    # Then it returns canonical bad-request error
    assert response.status_code == 400
    payload = response.json()
    assert payload["error_code"] == "CLIPS_TIME_RANGE_INVALID"


def test_list_clips_forwards_filter_values_without_route_normalization(tmp_path) -> None:
    """GET /clips should forward filter values as provided by the caller."""
    # Given an app with a call-recording repository
    manager = _write_config(tmp_path, cameras=[])
    repository = _StubRepository()
    app = _StubApp(
        config_manager=manager,
        repository=repository,
        storage=_StubStorage(),
    )
    client = _client(app)

    # When filtering with mixed-case values
    response = client.get("/api/v1/clips?risk_level=HIGH&activity_type=DELIVERY")

    # Then route forwards raw values to repository
    assert response.status_code == 200
    call = repository.list_clips_calls[-1]
    assert call["risk_level"] == "HIGH"
    assert call["activity_type"] == "DELIVERY"


def test_list_clips_forwards_status_alias_to_repository(tmp_path) -> None:
    """GET /clips should map query param status to repository status argument."""
    # Given an app with a call-recording repository
    manager = _write_config(tmp_path, cameras=[])
    repository = _StubRepository()
    app = _StubApp(
        config_manager=manager,
        repository=repository,
        storage=_StubStorage(),
    )
    client = _client(app)

    # When calling list with explicit status
    response = client.get("/api/v1/clips?status=deleted")

    # Then status is forwarded as ClipStatus enum
    assert response.status_code == 200
    assert repository.list_clips_calls[-1]["status"] == ClipStatus.DELETED


def test_list_clips_forwards_alerted_filter_to_repository(tmp_path) -> None:
    """GET /clips should forward the alerted filter to the repository."""
    # Given an app with a call-recording repository
    manager = _write_config(tmp_path, cameras=[])
    repository = _StubRepository()
    app = _StubApp(
        config_manager=manager,
        repository=repository,
        storage=_StubStorage(),
    )
    client = _client(app)

    # When filtering for alerted=false
    response = client.get("/api/v1/clips?alerted=false")

    # Then alerted filter is forwarded as a boolean
    assert response.status_code == 200
    assert repository.list_clips_calls[-1]["alerted"] is False


def test_list_clips_forwards_detected_filter_to_repository(tmp_path) -> None:
    """GET /clips should forward the detected filter to the repository."""
    # Given an app with a call-recording repository
    manager = _write_config(tmp_path, cameras=[])
    repository = _StubRepository()
    app = _StubApp(
        config_manager=manager,
        repository=repository,
        storage=_StubStorage(),
    )
    client = _client(app)

    # When filtering for detected=true
    response = client.get("/api/v1/clips?detected=true")

    # Then detected filter is forwarded as a boolean
    assert response.status_code == 200
    assert repository.list_clips_calls[-1]["detected"] is True


def test_list_clips_invalid_query_returns_canonical_validation_error(tmp_path) -> None:
    """GET /clips should return canonical envelope for query validation failures."""
    # Given a healthy app and authenticated clips endpoint
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(),
        storage=_StubStorage(),
    )
    client = _client(app)

    # When requesting clips with an invalid limit value
    response = client.get("/api/v1/clips?limit=0")

    # Then it returns canonical validation envelope with error details
    assert response.status_code == 422
    payload = response.json()
    assert payload["detail"] == "Request validation failed"
    assert payload["error_code"] == "REQUEST_VALIDATION_FAILED"
    assert isinstance(payload["validation_errors"], list)
    assert payload["validation_errors"]


def test_get_clip_includes_analysis_and_alert_details(tmp_path) -> None:
    """GET /clips/{id} should include analysis, detection, and alert fields."""
    # Given a clip with analysis and alert details
    created_at = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    clip = ClipStateData(
        clip_id="clip-1",
        camera_name="front",
        status=ClipStatus.ANALYZED,
        local_path="/tmp/clip-1.mp4",
        storage_uri="dropbox://clip-1.mp4",
        view_url="https://example/view",
        created_at=created_at,
        filter_result=FilterResult(
            detected_classes=["person", "package"],
            confidence=0.91,
            model="yolo",
            sampled_frames=12,
        ),
        analysis_result=AnalysisResult(
            risk_level=RiskLevel.HIGH,
            activity_type="delivery",
            summary="Package drop",
        ),
        alert_decision=AlertDecision(
            notify=True,
            notify_reason="risk_level=high",
        ),
    )
    manager = _write_config(tmp_path, cameras=[])
    repository = _StubRepository(clips=[clip])
    app = _StubApp(config_manager=manager, repository=repository, storage=_StubStorage())
    client = _client(app)

    # When requesting the clip
    response = client.get("/api/v1/clips/clip-1")

    # Then it returns detailed clip data
    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "clip-1"
    assert payload["camera"] == "front"
    assert payload["status"] == "analyzed"
    returned_created_at = dt.datetime.fromisoformat(payload["created_at"].replace("Z", "+00:00"))
    assert returned_created_at == created_at
    assert payload["detected_objects"] == ["person", "package"]
    assert payload["activity_type"] == "delivery"
    assert payload["risk_level"] == "high"
    assert payload["summary"] == "Package drop"
    assert payload["alerted"] is True
    assert payload["storage_uri"] == "dropbox://clip-1.mp4"
    assert payload["view_url"] == "https://example/view"


def test_get_clip_missing_returns_404(tmp_path) -> None:
    """GET /clips/{id} should return 404 when missing."""
    # Given an empty repository
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())
    client = _client(app)

    # When requesting a missing clip
    response = client.get("/api/v1/clips/missing")

    # Then it returns 404 with canonical error code
    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "CLIP_NOT_FOUND"


def test_get_clip_media_missing_returns_404(tmp_path) -> None:
    """GET /clips/{id}/media should return 404 when clip is missing."""
    # Given an empty repository
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())
    client = _client(app)

    # When requesting media for a missing clip
    response = client.get("/api/v1/clips/missing/media")

    # Then it returns 404 with canonical clip error
    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "CLIP_NOT_FOUND"


def test_get_clip_media_missing_storage_uri_returns_409(tmp_path) -> None:
    """GET /clips/{id}/media should return 409 when media is unavailable."""
    # Given a clip state without storage URI
    clip = ClipStateData(
        clip_id="clip-1",
        camera_name="front",
        status=ClipStatus.QUEUED_LOCAL,
        local_path="/tmp/clip-1.mp4",
    )
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(clips=[clip]),
        storage=_StubStorage(),
    )
    client = _client(app)

    # When requesting media for the clip
    response = client.get("/api/v1/clips/clip-1/media")

    # Then it returns canonical media-unavailable error
    assert response.status_code == 409
    payload = response.json()
    assert payload["error_code"] == "CLIP_MEDIA_UNAVAILABLE"


def test_get_clip_media_proxy_success_returns_inline_video(tmp_path) -> None:
    """GET /clips/{id}/media should proxy media inline for playback."""
    # Given a clip with storage URI and downloadable media
    clip = ClipStateData(
        clip_id="clip-1",
        camera_name="front",
        status=ClipStatus.UPLOADED,
        local_path="/tmp/clip-1.mp4",
        storage_uri="dropbox:/clips/clip-1.mp4",
    )
    manager = _write_config(tmp_path, cameras=[])
    storage = _StubStorage(media_bytes=b"video-bytes")
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(clips=[clip]),
        storage=storage,
    )
    client = _client(app)

    # When requesting clip media
    response = client.get("/api/v1/clips/clip-1/media")

    # Then it returns inline media payload for in-app playback
    assert response.status_code == 200
    assert response.content == b"video-bytes"
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["content-disposition"] == 'inline; filename="clip-1.mp4"'


def test_get_clip_media_defaults_filename_suffix_when_storage_uri_has_no_extension(
    tmp_path,
) -> None:
    """GET /clips/{id}/media should default filename suffix to .mp4 when storage URI lacks one."""
    # Given a clip with storage URI path missing a file extension
    clip = ClipStateData(
        clip_id="clip-1",
        camera_name="front",
        status=ClipStatus.UPLOADED,
        local_path="/tmp/clip-1.mp4",
        storage_uri="dropbox:/clips/clip-1",
    )
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(clips=[clip]),
        storage=_StubStorage(media_bytes=b"video-bytes"),
    )
    client = _client(app)

    # When requesting clip media
    response = client.get("/api/v1/clips/clip-1/media")

    # Then route serves content with default .mp4 filename
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'inline; filename="clip-1.mp4"'


def test_get_clip_media_storage_failure_returns_502(tmp_path) -> None:
    """GET /clips/{id}/media should return 502 when storage fetch fails."""
    # Given a clip with storage URI but failing storage backend
    clip = ClipStateData(
        clip_id="clip-1",
        camera_name="front",
        status=ClipStatus.UPLOADED,
        local_path="/tmp/clip-1.mp4",
        storage_uri="dropbox:/clips/clip-1.mp4",
    )
    manager = _write_config(tmp_path, cameras=[])
    storage = _StubStorage(fail_get=True)
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(clips=[clip]),
        storage=storage,
    )
    client = _client(app)

    # When requesting clip media
    response = client.get("/api/v1/clips/clip-1/media")

    # Then it returns canonical upstream fetch failure
    assert response.status_code == 502
    payload = response.json()
    assert payload["error_code"] == "CLIP_MEDIA_FETCH_FAILED"


def test_get_clip_media_success_cleans_temp_directory(tmp_path, monkeypatch) -> None:
    """GET /clips/{id}/media should clean temporary media files after response."""
    # Given a clip with storage URI and deterministic temp directory
    clip = ClipStateData(
        clip_id="clip-1",
        camera_name="front",
        status=ClipStatus.UPLOADED,
        local_path="/tmp/clip-1.mp4",
        storage_uri="dropbox:/clips/clip-1.mp4",
    )
    manager = _write_config(tmp_path, cameras=[])
    storage = _StubStorage(media_bytes=b"video-bytes")
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(clips=[clip]),
        storage=storage,
    )
    client = _client(app)
    temp_dir = tmp_path / "media-temp-dir"

    def _mkdtemp(*, prefix: str) -> str:
        _ = prefix
        temp_dir.mkdir(parents=True, exist_ok=True)
        return str(temp_dir)

    monkeypatch.setattr("homesec.api.routes.media.tempfile.mkdtemp", _mkdtemp)

    # When requesting clip media
    response = client.get("/api/v1/clips/clip-1/media")

    # Then response succeeds and temp directory is removed
    assert response.status_code == 200
    assert not temp_dir.exists()


def test_create_clip_media_token_auth_disabled_returns_direct_media_url(tmp_path) -> None:
    """POST /clips/{id}/media-token should return direct media path when auth is disabled."""
    # Given a clip with storage available and auth disabled
    clip = ClipStateData(
        clip_id="clip-1",
        camera_name="front",
        status=ClipStatus.UPLOADED,
        local_path="/tmp/clip-1.mp4",
        storage_uri="dropbox:/clips/clip-1.mp4",
    )
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(clips=[clip]),
        storage=_StubStorage(),
        server_config=FastAPIServerConfig(auth_enabled=False),
    )
    client = _client(app)

    # When requesting media token metadata
    response = client.post("/api/v1/clips/clip-1/media-token")

    # Then it returns a direct /media URL without token metadata
    assert response.status_code == 200
    payload = response.json()
    assert payload["media_url"] == "/api/v1/clips/clip-1/media"
    assert payload["tokenized"] is False
    assert payload["expires_at"] is None


def test_create_clip_media_token_missing_clip_returns_404(tmp_path) -> None:
    """POST /clips/{id}/media-token should return 404 when clip is missing."""
    # Given no clip state exists for requested id
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())
    client = _client(app)

    # When requesting media token metadata
    response = client.post("/api/v1/clips/missing/media-token")

    # Then it returns canonical clip-not-found error
    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "CLIP_NOT_FOUND"


def test_create_clip_media_token_missing_storage_uri_returns_409(tmp_path) -> None:
    """POST /clips/{id}/media-token should return 409 when media is unavailable."""
    # Given clip exists but has not been uploaded to storage
    clip = ClipStateData(
        clip_id="clip-1",
        camera_name="front",
        status=ClipStatus.QUEUED_LOCAL,
        local_path="/tmp/clip-1.mp4",
    )
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(clips=[clip]),
        storage=_StubStorage(),
    )
    client = _client(app)

    # When requesting media token metadata
    response = client.post("/api/v1/clips/clip-1/media-token")

    # Then it returns canonical media-unavailable error
    assert response.status_code == 409
    payload = response.json()
    assert payload["error_code"] == "CLIP_MEDIA_UNAVAILABLE"


def test_get_clip_media_accepts_api_key_when_auth_enabled(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /clips/{id}/media should allow API-key authenticated playback."""
    # Given auth is enabled with valid API key and media available
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    clip = ClipStateData(
        clip_id="clip-1",
        camera_name="front",
        status=ClipStatus.UPLOADED,
        local_path="/tmp/clip-1.mp4",
        storage_uri="dropbox:/clips/clip-1.mp4",
    )
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(clips=[clip]),
        storage=_StubStorage(media_bytes=b"video-bytes"),
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY"),
    )
    client = _client(app)

    # When requesting media with a valid API key header
    response = client.get(
        "/api/v1/clips/clip-1/media",
        headers={"Authorization": "Bearer secret"},
    )

    # Then playback succeeds without a token query parameter
    assert response.status_code == 200
    assert response.content == b"video-bytes"


def test_get_clip_media_rejects_invalid_token_when_auth_enabled(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GET /clips/{id}/media should reject invalid tokens when auth is enabled."""
    # Given auth enabled and a valid clip state
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    clip = ClipStateData(
        clip_id="clip-1",
        camera_name="front",
        status=ClipStatus.UPLOADED,
        local_path="/tmp/clip-1.mp4",
        storage_uri="dropbox:/clips/clip-1.mp4",
    )
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(clips=[clip]),
        storage=_StubStorage(media_bytes=b"video-bytes"),
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY"),
    )
    client = _client(app)

    # When requesting media with an invalid token and no API key header
    response = client.get("/api/v1/clips/clip-1/media?token=invalid-token")

    # Then it returns canonical media token rejection
    assert response.status_code == 401
    payload = response.json()
    assert payload["error_code"] == "MEDIA_TOKEN_REJECTED"


def test_get_clip_media_rejects_missing_token_when_auth_enabled(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GET /clips/{id}/media should reject unauthenticated requests without token query parameter."""
    # Given auth enabled and a clip available for playback
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    clip = ClipStateData(
        clip_id="clip-1",
        camera_name="front",
        status=ClipStatus.UPLOADED,
        local_path="/tmp/clip-1.mp4",
        storage_uri="dropbox:/clips/clip-1.mp4",
    )
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(clips=[clip]),
        storage=_StubStorage(media_bytes=b"video-bytes"),
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY"),
    )
    client = _client(app)

    # When requesting media without Authorization header and without token query parameter
    response = client.get("/api/v1/clips/clip-1/media")

    # Then request is rejected with canonical media-token error
    assert response.status_code == 401
    payload = response.json()
    assert payload["error_code"] == "MEDIA_TOKEN_REJECTED"


def test_get_clip_media_accepts_valid_media_token_when_auth_enabled(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GET /clips/{id}/media should allow token-authenticated browser playback."""
    # Given auth enabled with a clip and API key
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    clip = ClipStateData(
        clip_id="clip-1",
        camera_name="front",
        status=ClipStatus.UPLOADED,
        local_path="/tmp/clip-1.mp4",
        storage_uri="dropbox:/clips/clip-1.mp4",
    )
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(clips=[clip]),
        storage=_StubStorage(media_bytes=b"video-bytes"),
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY"),
    )
    client = _client(app)

    # When minting a media token with API key auth
    mint_response = client.post(
        "/api/v1/clips/clip-1/media-token",
        headers={"Authorization": "Bearer secret"},
    )

    # Then minting succeeds and returns tokenized media URL
    assert mint_response.status_code == 200
    mint_payload = mint_response.json()
    media_url = mint_payload["media_url"]
    assert mint_payload["tokenized"] is True
    parsed_url = urlparse(media_url)
    query = parse_qs(parsed_url.query)
    assert "token" in query
    assert query["token"]

    # When requesting media with token and no API key header
    response = client.get(media_url)

    # Then playback succeeds for browser-style token access
    assert response.status_code == 200
    assert response.content == b"video-bytes"


def test_create_clip_media_token_rejects_token_only_auth(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /clips/{id}/media-token should reject token-only access to prevent token chaining."""
    # Given auth enabled with a clip and valid API key
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    clip = ClipStateData(
        clip_id="clip-1",
        camera_name="front",
        status=ClipStatus.UPLOADED,
        local_path="/tmp/clip-1.mp4",
        storage_uri="dropbox:/clips/clip-1.mp4",
    )
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(clips=[clip]),
        storage=_StubStorage(media_bytes=b"video-bytes"),
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY"),
    )
    client = _client(app)

    # When minting once with API key to obtain a media token
    first_mint = client.post(
        "/api/v1/clips/clip-1/media-token",
        headers={"Authorization": "Bearer secret"},
    )
    assert first_mint.status_code == 200
    tokenized_url = first_mint.json()["media_url"]
    token = parse_qs(urlparse(tokenized_url).query)["token"][0]

    # When trying to mint again using only media token and no API key header
    second_mint = client.post(f"/api/v1/clips/clip-1/media-token?token={token}")

    # Then request is rejected because mint endpoint requires API key auth
    assert second_mint.status_code == 401
    payload = second_mint.json()
    assert payload["error_code"] == "UNAUTHORIZED"


def test_delete_clip_storage_failure_returns_500(tmp_path) -> None:
    """DELETE /clips/{id} should return 500 if storage deletion fails."""
    # Given a clip with a storage URI
    clip = ClipStateData(
        clip_id="clip-1",
        camera_name="front",
        status="uploaded",
        local_path="/tmp/clip-1.mp4",
        storage_uri="dropbox://clip-1.mp4",
    )
    manager = _write_config(tmp_path, cameras=[])
    repository = _StubRepository(clips=[clip])
    storage = _StubStorage(fail_delete=True)
    app = _StubApp(config_manager=manager, repository=repository, storage=storage)
    client = _client(app)

    # When deleting the clip
    response = client.delete("/api/v1/clips/clip-1")

    # Then it returns 500 after DB state has already been marked deleted
    assert response.status_code == 500
    payload = response.json()
    assert payload["error_code"] == "CLIP_STORAGE_DELETE_FAILED"
    assert repository.deleted_clip_ids == ["clip-1"]


def test_delete_clip_missing_returns_404(tmp_path) -> None:
    """DELETE /clips/{id} should return 404 when missing."""
    # Given an empty repository
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())
    client = _client(app)

    # When deleting a missing clip
    response = client.delete("/api/v1/clips/missing")

    # Then it returns 404 with canonical error code
    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "CLIP_NOT_FOUND"


def test_delete_clip_success_removes_storage(tmp_path) -> None:
    """DELETE /clips/{id} should return clip data on success."""
    # Given a clip stored in storage
    clip = ClipStateData(
        clip_id="clip-2",
        camera_name="front",
        status="uploaded",
        local_path="/tmp/clip-2.mp4",
        storage_uri="dropbox://clip-2.mp4",
    )
    manager = _write_config(tmp_path, cameras=[])
    repository = _StubRepository(clips=[clip])

    class _TrackingStorage(_StubStorage):
        def __init__(self) -> None:
            super().__init__(ok=True, fail_delete=False)
            self.deleted: list[str] = []

        async def delete(self, storage_uri: str) -> None:
            self.deleted.append(storage_uri)
            await super().delete(storage_uri)

    storage = _TrackingStorage()
    app = _StubApp(config_manager=manager, repository=repository, storage=storage)
    client = _client(app)

    # When deleting the clip
    response = client.delete("/api/v1/clips/clip-2")

    # Then it returns clip data and deletes storage
    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "clip-2"
    assert payload["storage_uri"] == "dropbox://clip-2.mp4"
    assert storage.deleted == ["dropbox://clip-2.mp4"]
    assert repository.deleted_clip_ids == ["clip-2"]


def test_delete_clip_returns_404_when_repository_delete_races(tmp_path) -> None:
    """DELETE /clips/{id} should return 404 when repository delete fails after initial read."""
    # Given clip exists for initial read but repository delete raises ValueError
    clip = ClipStateData(
        clip_id="clip-1",
        camera_name="front",
        status=ClipStatus.UPLOADED,
        local_path="/tmp/clip-1.mp4",
        storage_uri="dropbox:/clips/clip-1.mp4",
    )

    class _RaceDeleteRepository(_StubRepository):
        async def delete_clip(self, clip_id: str) -> ClipStateData:
            _ = clip_id
            raise ValueError("Clip not found: clip-1")

    storage = _StubStorage()
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_RaceDeleteRepository(clips=[clip]),
        storage=storage,
    )
    client = _client(app)

    # When deleting the clip
    response = client.delete("/api/v1/clips/clip-1")

    # Then it returns canonical clip-not-found response
    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "CLIP_NOT_FOUND"
    assert storage.delete_calls == []


def test_delete_clip_returns_500_when_repository_delete_fails_unexpectedly(tmp_path) -> None:
    """DELETE /clips/{id} should map unexpected repository failures to canonical 500."""
    # Given clip exists for initial read but delete operation fails unexpectedly
    clip = ClipStateData(
        clip_id="clip-1",
        camera_name="front",
        status=ClipStatus.UPLOADED,
        local_path="/tmp/clip-1.mp4",
        storage_uri="dropbox:/clips/clip-1.mp4",
    )

    class _BrokenDeleteRepository(_StubRepository):
        async def delete_clip(self, clip_id: str) -> ClipStateData:
            _ = clip_id
            raise RuntimeError("database write failed")

    storage = _StubStorage()
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_BrokenDeleteRepository(clips=[clip]),
        storage=storage,
    )
    client = _client(app)

    # When deleting the clip
    response = client.delete("/api/v1/clips/clip-1")

    # Then it returns canonical 500 and does not attempt storage deletion
    assert response.status_code == 500
    payload = response.json()
    assert payload["error_code"] == "CLIP_DELETE_MARK_FAILED"
    assert storage.delete_calls == []


def test_delete_clip_restores_storage_uri_when_repository_clears_it(tmp_path) -> None:
    """DELETE /clips/{id} should preserve storage URI in response when repository omits it."""
    # Given repository delete result drops storage_uri unexpectedly
    original = ClipStateData(
        clip_id="clip-1",
        camera_name="front",
        status=ClipStatus.UPLOADED,
        local_path="/tmp/clip-1.mp4",
        storage_uri="dropbox:/clips/clip-1.mp4",
    )
    deleted = ClipStateData(
        clip_id="clip-1",
        camera_name="front",
        status=ClipStatus.DELETED,
        local_path="/tmp/clip-1.mp4",
        storage_uri=None,
    )

    class _ClearingDeleteRepository(_StubRepository):
        async def delete_clip(self, clip_id: str) -> ClipStateData:
            _ = clip_id
            return deleted

    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_ClearingDeleteRepository(clips=[original]),
        storage=_StubStorage(),
    )
    client = _client(app)

    # When deleting the clip
    response = client.delete("/api/v1/clips/clip-1")

    # Then response preserves original storage URI for client continuity
    assert response.status_code == 200
    payload = response.json()
    assert payload["storage_uri"] == "dropbox:/clips/clip-1.mp4"


def test_stats_endpoint_returns_counts(tmp_path) -> None:
    """GET /stats should return clip and alert counts."""
    # Given a repository with known counts
    manager = _write_config(tmp_path, cameras=[])
    repository = _StubRepository(clips_count=7, alerts_count=2)
    app = _StubApp(config_manager=manager, repository=repository, storage=_StubStorage())
    client = _client(app)

    # When requesting stats
    response = client.get("/api/v1/stats")

    # Then it returns the counts
    assert response.status_code == 200
    payload = response.json()
    assert payload["clips_today"] == 7
    assert payload["alerts_today"] == 2


def test_stats_endpoint_includes_camera_counts_and_uptime(tmp_path) -> None:
    """GET /stats should include camera totals and uptime."""
    # Given an app with two cameras and one healthy source
    manager = _write_config(
        tmp_path,
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp"}},
            },
            {
                "name": "back",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp"}},
            },
        ],
    )
    sources = {
        "front": _StubSource(healthy=True),
        "back": _StubSource(healthy=False),
    }
    repository = _StubRepository(clips_count=0, alerts_count=0)
    app = _StubApp(
        config_manager=manager,
        repository=repository,
        storage=_StubStorage(),
        sources_by_name=sources,
    )
    app.uptime_seconds = 12.5
    client = _client(app)

    # When requesting stats
    response = client.get("/api/v1/stats")

    # Then it returns camera totals and uptime
    assert response.status_code == 200
    payload = response.json()
    assert payload["cameras_total"] == 2
    assert payload["cameras_online"] == 1
    assert payload["uptime_seconds"] == 12.5


def test_system_restart_endpoint_removed_returns_404(tmp_path) -> None:
    """POST /system/restart should be removed from the API surface."""
    # Given a running app with API routes registered
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(),
        storage=_StubStorage(),
    )
    client = _client(app)

    # When requesting deprecated system restart endpoint
    response = client.post("/api/v1/system/restart")

    # Then endpoint is not found with canonical error code
    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "NOT_FOUND"


def test_db_unavailable_returns_503(tmp_path) -> None:
    """DB-backed endpoints should return 503 when DB is down."""
    # Given a repository that is unavailable
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(ok=False),
        storage=_StubStorage(),
    )
    client = _client(app)

    # When requesting a DB-backed endpoint
    response = client.get("/api/v1/clips")

    # Then it returns 503 with error code
    assert response.status_code == 503
    payload = response.json()
    assert payload["error_code"] == "DB_UNAVAILABLE"


def test_cameras_available_when_db_unavailable(tmp_path) -> None:
    """Camera config endpoint should remain available when DB is down."""
    # Given a repository that is unavailable
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(ok=False),
        storage=_StubStorage(),
    )
    client = _client(app)

    # When requesting camera config endpoint
    response = client.get("/api/v1/cameras")

    # Then it still returns successfully
    assert response.status_code == 200


def test_db_health_probe_is_cached_for_burst_requests(tmp_path) -> None:
    """DB-backed endpoints should reuse a recent health probe result."""
    # Given a healthy repository and DB-backed endpoint client
    manager = _write_config(tmp_path, cameras=[])
    repository = _StubRepository(ok=True)
    app = _StubApp(
        config_manager=manager,
        repository=repository,
        storage=_StubStorage(),
    )
    client = _client(app)

    # When issuing two immediate requests to a DB-backed endpoint
    first = client.get("/api/v1/clips")
    second = client.get("/api/v1/clips")

    # Then both requests succeed and the ping probe is reused
    assert first.status_code == 200
    assert second.status_code == 200
    assert repository.ping_calls == 1


def test_db_health_probe_rechecks_after_ttl_and_detects_outage(tmp_path) -> None:
    """DB-backed endpoints should re-check health after TTL and fail closed when DB drops."""
    # Given a repository that starts healthy and later becomes unavailable
    manager = _write_config(tmp_path, cameras=[])
    repository = _StubRepository(ok=True)
    app = _StubApp(
        config_manager=manager,
        repository=repository,
        storage=_StubStorage(),
    )
    client = _client(app)

    # When first request succeeds, then DB is flipped down after cache TTL expires
    first = client.get("/api/v1/clips")
    repository._ok = False  # Simulate DB outage after initial successful probe.
    time.sleep(0.6)
    second = client.get("/api/v1/clips")

    # Then second request rechecks DB and returns canonical outage error
    assert first.status_code == 200
    assert second.status_code == 503
    payload = second.json()
    assert payload["error_code"] == "DB_UNAVAILABLE"
    assert repository.ping_calls >= 2


def test_root_health_endpoint_matches_versioned_health(tmp_path) -> None:
    """GET /health should expose the same probe payload as /api/v1/health."""
    # Given a healthy app with one online camera
    manager = _write_config(
        tmp_path,
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp"}},
            }
        ],
    )
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(ok=True),
        storage=_StubStorage(),
        sources_by_name={"front": _StubSource(healthy=True, heartbeat=1.0)},
    )
    client = _client(app)

    # When requesting both health endpoints
    root_response = client.get("/health")
    versioned_response = client.get("/api/v1/health")

    # Then both return the same healthy payload
    assert root_response.status_code == 200
    assert versioned_response.status_code == 200
    assert root_response.json() == versioned_response.json()


def test_health_endpoints_degrade_when_postgres_unavailable(tmp_path) -> None:
    """Health endpoints should report degraded when pipeline runs but DB is unavailable."""
    # Given a running pipeline with DB unavailable
    manager = _write_config(
        tmp_path,
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp"}},
            }
        ],
    )
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(ok=False),
        storage=_StubStorage(),
        sources_by_name={"front": _StubSource(healthy=True, heartbeat=1.0)},
        pipeline_running=True,
    )
    client = _client(app)

    # When requesting health probe endpoints
    root_response = client.get("/health")
    versioned_response = client.get("/api/v1/health")

    # Then both report degraded health with 200 status
    assert root_response.status_code == 200
    assert versioned_response.status_code == 200
    assert root_response.json()["status"] == "degraded"
    assert root_response.json()["postgres"] == "unavailable"
    assert root_response.json() == versioned_response.json()


def test_health_endpoints_return_503_when_pipeline_stopped(tmp_path) -> None:
    """Health probe endpoints should return 503 when pipeline is stopped."""
    # Given a stopped pipeline
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(ok=True),
        storage=_StubStorage(),
        pipeline_running=False,
    )
    client = _client(app)

    # When requesting health probe endpoints
    root_response = client.get("/health")
    versioned_response = client.get("/api/v1/health")

    # Then both report unhealthy with 503 status
    assert root_response.status_code == 503
    assert versioned_response.status_code == 503
    assert root_response.json()["status"] == "unhealthy"
    assert root_response.json()["pipeline"] == "stopped"
    assert root_response.json() == versioned_response.json()


def test_diagnostics_reports_degraded_when_storage_ping_raises(tmp_path) -> None:
    """Diagnostics should degrade when component ping raises and preserve error detail."""

    class _RaisingStorage(_StubStorage):
        async def ping(self) -> bool:
            raise RuntimeError("storage ping boom")

    # Given a running pipeline with storage ping failures
    manager = _write_config(
        tmp_path,
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp"}},
            }
        ],
    )
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(ok=True),
        storage=_RaisingStorage(),
        sources_by_name={"front": _StubSource(healthy=True, heartbeat=1.0)},
        pipeline_running=True,
    )
    client = _client(app)

    # When requesting diagnostics
    response = client.get("/api/v1/diagnostics")

    # Then diagnostics degrades and surfaces storage component error state
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["storage"]["status"] == "error"
    assert payload["storage"]["error"] == "storage ping boom"


def test_diagnostics_ignores_disabled_unhealthy_camera_for_global_status(tmp_path) -> None:
    """Diagnostics should not degrade when only disabled cameras are unhealthy."""
    # Given a running pipeline where only disabled camera appears unhealthy
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(ok=True),
        storage=_StubStorage(),
        sources_by_name={},
        pipeline_running=True,
    )
    app._config.cameras = [  # type: ignore[attr-defined]
        CameraConfig(
            name="garage",
            enabled=False,
            source=CameraSourceConfig(backend="local_folder", config={"watch_dir": "/tmp"}),
        )
    ]
    client = _client(app)

    # When requesting diagnostics
    response = client.get("/api/v1/diagnostics")

    # Then diagnostics remains healthy and marks camera disabled with no heartbeat
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["cameras"]["garage"]["enabled"] is False
    assert payload["cameras"]["garage"]["healthy"] is False
    assert payload["cameras"]["garage"]["last_heartbeat"] is None


def test_diagnostics_degrades_when_enabled_camera_has_no_source(tmp_path) -> None:
    """Diagnostics should degrade when an enabled camera has no live source object."""
    # Given a running pipeline with an enabled camera that is missing its source
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(ok=True),
        storage=_StubStorage(),
        sources_by_name={},
        pipeline_running=True,
    )
    app._config.cameras = [  # type: ignore[attr-defined]
        CameraConfig(
            name="front",
            enabled=True,
            source=CameraSourceConfig(backend="local_folder", config={"watch_dir": "/tmp"}),
        )
    ]
    client = _client(app)

    # When requesting diagnostics
    response = client.get("/api/v1/diagnostics")

    # Then diagnostics degrades due to enabled camera health failure
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["cameras"]["front"]["enabled"] is True
    assert payload["cameras"]["front"]["healthy"] is False


def test_diagnostics_reports_unhealthy_when_pipeline_stopped(tmp_path) -> None:
    """Diagnostics should report unhealthy status when runtime pipeline is stopped."""
    # Given pipeline is stopped while components otherwise report healthy
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(ok=True),
        storage=_StubStorage(),
        sources_by_name={},
        pipeline_running=False,
    )
    client = _client(app)

    # When requesting diagnostics
    response = client.get("/api/v1/diagnostics")

    # Then diagnostics status is unhealthy
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unhealthy"


def test_auth_required_when_enabled(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Auth should be enforced for non-public endpoints."""
    # Given auth is enabled
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    server_config = FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY")
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(),
        storage=_StubStorage(),
        server_config=server_config,
    )
    client = _client(app)

    # When missing Authorization header
    response = client.get("/api/v1/cameras")

    # Then it returns 401
    assert response.status_code == 401
    payload = response.json()
    assert payload["error_code"] == "UNAUTHORIZED"

    # When using an incorrect token
    response = client.get("/api/v1/cameras", headers={"Authorization": "Bearer wrong"})

    # Then it returns 401
    assert response.status_code == 401
    payload = response.json()
    assert payload["error_code"] == "UNAUTHORIZED"

    # When using the correct token
    response = client.get("/api/v1/cameras", headers={"Authorization": "Bearer secret"})

    # Then it returns 200
    assert response.status_code == 200

    # When hitting a public endpoint without auth
    response = client.get("/api/v1/health")

    # Then it returns 200
    assert response.status_code == 200

    # When hitting the root health probe endpoint without auth
    response = client.get("/health")

    # Then it returns 200
    assert response.status_code == 200

    # When hitting diagnostics without auth
    response = client.get("/api/v1/diagnostics")

    # Then it returns 401
    assert response.status_code == 401
    payload = response.json()
    assert payload["error_code"] == "UNAUTHORIZED"

    # When hitting diagnostics with correct auth
    response = client.get("/api/v1/diagnostics", headers={"Authorization": "Bearer secret"})

    # Then it returns 200
    assert response.status_code == 200


def test_auth_env_missing_returns_500(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Auth should return 500 when API key is not configured."""
    # Given auth is enabled but env var missing
    monkeypatch.delenv("HOMESEC_API_KEY", raising=False)
    server_config = FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY")
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(
        config_manager=manager,
        repository=_StubRepository(),
        storage=_StubStorage(),
        server_config=server_config,
    )
    client = _client(app)

    # When requesting an authenticated endpoint
    response = client.get("/api/v1/cameras")

    # Then it returns 500 with canonical error code
    assert response.status_code == 500
    payload = response.json()
    assert payload["error_code"] == "API_KEY_NOT_CONFIGURED"
