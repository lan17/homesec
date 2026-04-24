"""Tests for maintenance API routes."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from fastapi.testclient import TestClient

from homesec.api.server import create_app
from homesec.maintenance.postgres_backup import PostgresBackupRunRequest, PostgresBackupStatus
from homesec.models.config import FastAPIServerConfig
from tests.homesec.ui_dist_stub import ensure_stub_ui_dist


class _StubBackupManager:
    def __init__(
        self,
        status: PostgresBackupStatus,
        request: PostgresBackupRunRequest | None = None,
    ) -> None:
        self._status = status
        self._request = request or PostgresBackupRunRequest(True, "Postgres backup accepted")
        self.request_count = 0

    def status(self) -> PostgresBackupStatus:
        return self._status

    def request_backup_now(self) -> PostgresBackupRunRequest:
        self.request_count += 1
        return self._request


class _StubMaintenanceApp:
    def __init__(self, manager: _StubBackupManager) -> None:
        self._manager = manager
        self._bootstrap_mode = False
        self._config = SimpleNamespace(
            server=ensure_stub_ui_dist(FastAPIServerConfig()),
            cameras=[],
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
    def postgres_backup_manager(self) -> _StubBackupManager:
        return self._manager


def _status(
    *,
    enabled: bool = True,
    running: bool = False,
    available: bool = True,
    reason: str | None = None,
) -> PostgresBackupStatus:
    now = dt.datetime(2026, 4, 23, 16, 0, tzinfo=dt.timezone.utc)
    return PostgresBackupStatus(
        enabled=enabled,
        running=running,
        available=available,
        unavailable_reason=reason,
        last_attempted_at=now,
        last_success_at=now,
        last_error=None,
        last_local_path="/backups/homesec-postgres-20260423-160000.dump",
        last_uploaded_uri="local:/storage/backups/homesec-postgres-20260423-160000.dump",
        next_run_at=now + dt.timedelta(hours=24),
        pending_remote_delete_count=1,
    )


def _client(manager: _StubBackupManager) -> TestClient:
    return TestClient(create_app(_StubMaintenanceApp(manager)))


def test_postgres_backup_status_endpoint_returns_manager_status() -> None:
    """GET backup status should expose manager status fields."""
    # Given: A backup manager with current status
    manager = _StubBackupManager(_status())
    client = _client(manager)

    # When: Requesting backup status
    response = client.get("/api/v1/maintenance/postgres-backups/status")

    # Then: Status fields are returned
    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["running"] is False
    assert payload["available"] is True
    assert payload["last_uploaded_uri"].startswith("local:")
    assert payload["pending_remote_delete_count"] == 1


def test_postgres_backup_run_endpoint_accepts_manual_backup() -> None:
    """POST backup run should return 202 when single-flight accepts the request."""
    # Given: Backups are enabled and available
    manager = _StubBackupManager(_status())
    client = _client(manager)

    # When: Triggering a manual backup
    response = client.post("/api/v1/maintenance/postgres-backups/run")

    # Then: The request is accepted
    assert response.status_code == 202
    assert response.json()["accepted"] is True
    assert manager.request_count == 1


def test_postgres_backup_run_endpoint_rejects_busy_manager() -> None:
    """POST backup run should return 409 when another backup is in flight."""
    # Given: The manager reports a running backup
    manager = _StubBackupManager(
        _status(running=True),
        PostgresBackupRunRequest(False, "Postgres backup already running"),
    )
    client = _client(manager)

    # When: Triggering another manual backup
    response = client.post("/api/v1/maintenance/postgres-backups/run")

    # Then: A structured conflict response is returned
    assert response.status_code == 409
    assert response.json()["error_code"] == "BACKUP_IN_PROGRESS"


def test_postgres_backup_run_endpoint_reports_disabled_or_unavailable() -> None:
    """POST backup run should return clear API errors for disabled/unavailable backups."""
    # Given: One app has backups disabled and another has pg_dump unavailable
    disabled_client = _client(_StubBackupManager(_status(enabled=False, available=False)))
    unavailable_client = _client(
        _StubBackupManager(_status(available=False, reason="pg_dump not found in PATH"))
    )

    # When: Triggering manual backups
    disabled_response = disabled_client.post("/api/v1/maintenance/postgres-backups/run")
    unavailable_response = unavailable_client.post("/api/v1/maintenance/postgres-backups/run")

    # Then: Stable error codes distinguish disabled from unavailable
    assert disabled_response.status_code == 503
    assert disabled_response.json()["error_code"] == "BACKUP_DISABLED"
    assert unavailable_response.status_code == 503
    assert unavailable_response.json()["error_code"] == "BACKUP_UNAVAILABLE"
