"""Tests for FastAPI camera and clip routes."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest
import yaml
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from pydantic import BaseModel

from homesec.api.server import create_app
from homesec.config.manager import ConfigManager
from homesec.models.alert import AlertDecision
from homesec.models.clip import ClipStateData
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
    ) -> None:
        self._clips = clips or []
        self._ok = ok
        self._clips_count = clips_count
        self._alerts_count = alerts_count
        self.deleted_clip_ids: list[str] = []
        self.ping_calls = 0

    async def ping(self) -> bool:
        self.ping_calls += 1
        return self._ok

    async def list_clips(
        self,
        *,
        camera: str | None = None,
        status: object | None = None,
        alerted: bool | None = None,
        risk_level: str | None = None,
        activity_type: str | None = None,
        since: dt.datetime | None = None,
        until: dt.datetime | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[ClipStateData], int]:
        _ = camera
        _ = status
        _ = alerted
        _ = risk_level
        _ = activity_type
        _ = since
        _ = until
        total = len(self._clips)
        return (self._clips[offset : offset + limit], total)

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
    def __init__(self, *, ok: bool = True, fail_delete: bool = False) -> None:
        self._ok = ok
        self._fail_delete = fail_delete

    async def ping(self) -> bool:
        return self._ok

    async def delete(self, storage_uri: str) -> None:
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


def test_create_camera_duplicate_returns_400(tmp_path) -> None:
    """POST /cameras should reject duplicate names."""
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

    # Then it returns a 400
    assert response.status_code == 400


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

    # Then it returns 404
    assert response.status_code == 404


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

    # Then it returns 404
    assert response.status_code == 404


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

    # Then it returns 400
    assert response.status_code == 400


def test_update_camera_missing_returns_404(tmp_path) -> None:
    """PUT /cameras/{name} should return 404 when missing."""
    # Given a config with no cameras
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())
    client = _client(app)

    # When updating a missing camera
    response = client.put("/api/v1/cameras/missing", json={"enabled": False})

    # Then it returns 404
    assert response.status_code == 404


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
                    },
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


def test_list_clips_pagination(tmp_path) -> None:
    """GET /clips should paginate results."""
    # Given 100 clips in the repository
    now = dt.datetime.now(dt.timezone.utc)
    clips = [
        ClipStateData(
            clip_id=f"clip-{idx}",
            camera_name="front",
            status="uploaded",
            local_path=f"/tmp/{idx}.mp4",
            created_at=now + dt.timedelta(seconds=idx),
        )
        for idx in range(100)
    ]
    manager = _write_config(tmp_path, cameras=[])
    repository = _StubRepository(clips=clips)
    app = _StubApp(config_manager=manager, repository=repository, storage=_StubStorage())
    client = _client(app)

    # When requesting page 2
    response = client.get("/api/v1/clips?page=2&page_size=10")

    # Then it returns the second page
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 100
    assert payload["page"] == 2
    assert payload["page_size"] == 10
    assert len(payload["clips"]) == 10


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

    # Then it returns 404
    assert response.status_code == 404


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

    # Then it returns 500
    assert response.status_code == 500
    assert repository.deleted_clip_ids == []


def test_delete_clip_missing_returns_404(tmp_path) -> None:
    """DELETE /clips/{id} should return 404 when missing."""
    # Given an empty repository
    manager = _write_config(tmp_path, cameras=[])
    app = _StubApp(config_manager=manager, repository=_StubRepository(), storage=_StubStorage())
    client = _client(app)

    # When deleting a missing clip
    response = client.delete("/api/v1/clips/missing")

    # Then it returns 404
    assert response.status_code == 404


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

    # Then endpoint is not found
    assert response.status_code == 404


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

    # When using an incorrect token
    response = client.get("/api/v1/cameras", headers={"Authorization": "Bearer wrong"})

    # Then it returns 401
    assert response.status_code == 401

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

    # Then it returns 500
    assert response.status_code == 500
