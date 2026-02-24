"""Tests for setup/onboarding service logic."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from homesec.models.config import FastAPIServerConfig
from homesec.services import setup as setup_service


class _StubRepository:
    def __init__(self, *, ok: bool = True) -> None:
        self._ok = ok

    async def ping(self) -> bool:
        return self._ok


class _StubApp:
    def __init__(
        self,
        *,
        bootstrap_mode: bool,
        pipeline_running: bool,
        has_cameras: bool,
        repository: _StubRepository | None,
        server_config: FastAPIServerConfig,
        clips_dir: str = "clips",
    ) -> None:
        self.bootstrap_mode = bootstrap_mode
        self.pipeline_running = pipeline_running
        self._repository = repository
        self._server_config = server_config
        if has_cameras:
            cameras = [SimpleNamespace(name="front")]
        else:
            cameras = []
        self._config = SimpleNamespace(
            cameras=cameras,
            server=server_config,
            storage=SimpleNamespace(paths=SimpleNamespace(clips_dir=clips_dir)),
        )

    @property
    def config(self):
        if self.bootstrap_mode:
            raise RuntimeError("Config not loaded")
        return self._config

    @property
    def repository(self):
        if self._repository is None:
            raise RuntimeError("Repository not initialized")
        return self._repository

    @property
    def server_config(self) -> FastAPIServerConfig:
        return self._server_config


def _server_config(
    *, auth_enabled: bool = False, api_key_env: str | None = None
) -> FastAPIServerConfig:
    return FastAPIServerConfig(auth_enabled=auth_enabled, api_key_env=api_key_env)


@pytest.mark.asyncio
async def test_get_setup_status_returns_fresh_in_bootstrap_mode() -> None:
    """Setup status should be fresh when no config is loaded in bootstrap mode."""
    # Given: A bootstrap-mode application with no config and no running pipeline
    app = _StubApp(
        bootstrap_mode=True,
        pipeline_running=False,
        has_cameras=False,
        repository=None,
        server_config=_server_config(auth_enabled=False),
    )

    # When: Reading setup status
    status = await setup_service.get_setup_status(app)

    # Then: Setup state is fresh with no cameras and no active pipeline
    assert status.state == "fresh"
    assert status.has_cameras is False
    assert status.pipeline_running is False
    assert status.auth_configured is True


@pytest.mark.asyncio
async def test_get_setup_status_returns_partial_without_pipeline() -> None:
    """Setup status should be partial when config exists but pipeline is not running."""
    # Given: A configured app with at least one camera but stopped pipeline
    app = _StubApp(
        bootstrap_mode=False,
        pipeline_running=False,
        has_cameras=True,
        repository=_StubRepository(ok=True),
        server_config=_server_config(auth_enabled=False),
    )

    # When: Reading setup status
    status = await setup_service.get_setup_status(app)

    # Then: Setup state is partial
    assert status.state == "partial"
    assert status.has_cameras is True
    assert status.pipeline_running is False


@pytest.mark.asyncio
async def test_get_setup_status_returns_complete_when_pipeline_running() -> None:
    """Setup status should be complete when cameras exist and pipeline runs."""
    # Given: A configured app with cameras and active pipeline
    app = _StubApp(
        bootstrap_mode=False,
        pipeline_running=True,
        has_cameras=True,
        repository=_StubRepository(ok=True),
        server_config=_server_config(auth_enabled=False),
    )

    # When: Reading setup status
    status = await setup_service.get_setup_status(app)

    # Then: Setup state is complete
    assert status.state == "complete"
    assert status.has_cameras is True
    assert status.pipeline_running is True


@pytest.mark.asyncio
async def test_get_setup_status_marks_auth_unconfigured_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setup status should report auth_configured false when auth key is missing."""
    # Given: Auth-enabled server config with missing API key env value
    monkeypatch.delenv("HOMESEC_TEST_API_KEY", raising=False)
    app = _StubApp(
        bootstrap_mode=True,
        pipeline_running=False,
        has_cameras=False,
        repository=None,
        server_config=_server_config(auth_enabled=True, api_key_env="HOMESEC_TEST_API_KEY"),
    )

    # When: Reading setup status
    status = await setup_service.get_setup_status(app)

    # Then: Auth is marked unconfigured
    assert status.auth_configured is False


@pytest.mark.asyncio
async def test_run_preflight_returns_all_passed_when_checks_succeed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should report all_passed when every check succeeds."""
    # Given: A configured app and deterministic successful probe helpers
    app = _StubApp(
        bootstrap_mode=False,
        pipeline_running=False,
        has_cameras=True,
        repository=_StubRepository(ok=True),
        server_config=_server_config(auth_enabled=False),
    )
    monkeypatch.setattr(setup_service, "shutil_which_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        setup_service,
        "shutil_disk_usage",
        lambda path: (10_000_000_000, 2_000_000_000, 8_000_000_000),
    )
    monkeypatch.setattr(setup_service, "_network_probe", lambda: (True, "DNS resolution succeeded"))

    # When: Running setup preflight
    response = await setup_service.run_preflight(app)

    # Then: Every check passes and aggregate result is successful
    assert response.all_passed is True
    by_name = {check.name: check for check in response.checks}
    assert by_name["postgres"].passed is True
    assert by_name["ffmpeg"].passed is True
    assert by_name["disk_space"].passed is True
    assert by_name["network"].passed is True


@pytest.mark.asyncio
async def test_run_preflight_reports_postgres_not_configured_in_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should fail postgres check when repository is not initialized."""
    # Given: A bootstrap app without repository and deterministic non-postgres probe success
    app = _StubApp(
        bootstrap_mode=True,
        pipeline_running=False,
        has_cameras=False,
        repository=None,
        server_config=_server_config(auth_enabled=False),
    )
    monkeypatch.setattr(setup_service, "shutil_which_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        setup_service,
        "shutil_disk_usage",
        lambda path: (10_000_000_000, 2_000_000_000, 8_000_000_000),
    )
    monkeypatch.setattr(setup_service, "_network_probe", lambda: (True, "DNS resolution succeeded"))

    # When: Running setup preflight
    response = await setup_service.run_preflight(app)

    # Then: Postgres reports not configured while other checks pass
    by_name = {check.name: check for check in response.checks}
    assert by_name["postgres"].passed is False
    assert by_name["postgres"].message == "Database not configured"
    assert by_name["ffmpeg"].passed is True
    assert by_name["disk_space"].passed is True
    assert by_name["network"].passed is True
    assert response.all_passed is False


def test_disk_probe_path_walks_up_to_existing_parent(tmp_path: Path) -> None:
    """Disk probe path should resolve to nearest existing ancestor for missing clips dir."""
    # Given: A configured app with nested non-existent clips directory
    nested = tmp_path / "missing" / "deep" / "clips"
    app = _StubApp(
        bootstrap_mode=False,
        pipeline_running=False,
        has_cameras=False,
        repository=None,
        server_config=_server_config(auth_enabled=False),
        clips_dir=str(nested),
    )

    # When: Resolving disk probe path
    path = setup_service._disk_probe_path(app)

    # Then: Service falls back to nearest existing ancestor
    assert path == tmp_path
