"""Tests for FastAPI health endpoints."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from homesec.api.server import create_app
from homesec.models.config import CameraConfig, CameraSourceConfig, FastAPIServerConfig


class _StubRepository:
    def __init__(self, ok: bool) -> None:
        self._ok = ok

    async def ping(self) -> bool:
        return self._ok


class _StubStorage:
    def __init__(self, ok: bool = True) -> None:
        self._ok = ok

    async def ping(self) -> bool:
        return self._ok

    async def delete(self, storage_uri: str) -> None:
        _ = storage_uri
        return None


class _StubSource:
    def __init__(self, healthy: bool = True, heartbeat: float = 1.0) -> None:
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
        repository: _StubRepository,
        storage: _StubStorage,
        sources_by_name: dict[str, _StubSource],
        pipeline_running: bool,
        cameras: list[CameraConfig] | None = None,
    ) -> None:
        if cameras is None:
            cameras = [
                CameraConfig(
                    name=name,
                    enabled=True,
                    source=CameraSourceConfig(
                        backend="local_folder",
                        config={"watch_dir": "/tmp"},
                    ),
                )
                for name in sources_by_name
            ]
        self._config = SimpleNamespace(
            server=FastAPIServerConfig(),
            cameras=cameras,
        )
        self.repository = repository
        self.storage = storage
        self.sources = list(sources_by_name.values())
        self._sources_by_name = sources_by_name
        self._pipeline_running = pipeline_running
        self.uptime_seconds = 123.0

    @property
    def config(self):  # type: ignore[override]
        return self._config

    @property
    def pipeline_running(self) -> bool:
        return self._pipeline_running

    def get_source(self, camera_name: str) -> _StubSource | None:
        return self._sources_by_name.get(camera_name)


def _client(app: _StubApp) -> TestClient:
    return TestClient(create_app(app))


def test_health_healthy() -> None:
    """Health should be healthy when pipeline and DB are up."""
    # Given a running pipeline with healthy DB and camera
    app = _StubApp(
        repository=_StubRepository(ok=True),
        storage=_StubStorage(ok=True),
        sources_by_name={"front": _StubSource(healthy=True)},
        pipeline_running=True,
    )
    client = _client(app)

    # When requesting health
    response = client.get("/api/v1/health")

    # Then it reports healthy
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["postgres"] == "connected"
    assert payload["pipeline"] == "running"
    assert payload["cameras_online"] == 1


def test_health_degraded_when_db_down() -> None:
    """Health should be degraded when DB is down."""
    # Given a running pipeline with DB down
    app = _StubApp(
        repository=_StubRepository(ok=False),
        storage=_StubStorage(ok=True),
        sources_by_name={"front": _StubSource(healthy=True)},
        pipeline_running=True,
    )
    client = _client(app)

    # When requesting health
    response = client.get("/api/v1/health")

    # Then it reports degraded
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["postgres"] == "unavailable"


def test_health_unhealthy_when_pipeline_stopped() -> None:
    """Health should be unhealthy when pipeline is stopped."""
    # Given a stopped pipeline
    app = _StubApp(
        repository=_StubRepository(ok=True),
        storage=_StubStorage(ok=True),
        sources_by_name={"front": _StubSource(healthy=True)},
        pipeline_running=False,
    )
    client = _client(app)

    # When requesting health
    response = client.get("/api/v1/health")

    # Then it returns 503 unhealthy
    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "unhealthy"
    assert payload["pipeline"] == "stopped"


def test_diagnostics_reports_camera_status() -> None:
    """Diagnostics should include per-camera health."""
    # Given a running pipeline with one camera
    app = _StubApp(
        repository=_StubRepository(ok=True),
        storage=_StubStorage(ok=True),
        sources_by_name={"front": _StubSource(healthy=True, heartbeat=42.0)},
        pipeline_running=True,
    )
    client = _client(app)

    # When requesting diagnostics
    response = client.get("/api/v1/diagnostics")

    # Then camera status is included
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["cameras"]["front"]["healthy"] is True
    assert payload["cameras"]["front"]["last_heartbeat"] == 42.0


def test_diagnostics_degraded_when_camera_unhealthy() -> None:
    """Diagnostics should be degraded when a camera is unhealthy."""
    # Given a running pipeline with an unhealthy camera
    app = _StubApp(
        repository=_StubRepository(ok=True),
        storage=_StubStorage(ok=True),
        sources_by_name={"front": _StubSource(healthy=False, heartbeat=10.0)},
        pipeline_running=True,
    )
    client = _client(app)

    # When requesting diagnostics
    response = client.get("/api/v1/diagnostics")

    # Then it reports degraded and camera is unhealthy
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["cameras"]["front"]["healthy"] is False


def test_diagnostics_degraded_when_storage_down_and_camera_disabled() -> None:
    """Diagnostics should be degraded when storage is unavailable."""
    # Given a running pipeline with storage down and a disabled camera
    camera = CameraConfig(
        name="front",
        enabled=False,
        source=CameraSourceConfig(
            backend="local_folder",
            config={"watch_dir": "/tmp"},
        ),
    )
    app = _StubApp(
        repository=_StubRepository(ok=True),
        storage=_StubStorage(ok=False),
        sources_by_name={},
        pipeline_running=True,
        cameras=[camera],
    )
    client = _client(app)

    # When requesting diagnostics
    response = client.get("/api/v1/diagnostics")

    # Then it reports degraded and camera is offline
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["storage"]["status"] == "error"
    assert payload["cameras"]["front"]["healthy"] is False
    assert payload["cameras"]["front"]["last_heartbeat"] is None


def test_diagnostics_unhealthy_when_pipeline_stopped() -> None:
    """Diagnostics should be unhealthy when pipeline is stopped."""
    # Given a stopped pipeline
    app = _StubApp(
        repository=_StubRepository(ok=True),
        storage=_StubStorage(ok=True),
        sources_by_name={"front": _StubSource(healthy=True)},
        pipeline_running=False,
    )
    client = _client(app)

    # When requesting diagnostics
    response = client.get("/api/v1/diagnostics")

    # Then it reports unhealthy
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unhealthy"
