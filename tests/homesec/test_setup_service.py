"""Tests for setup/onboarding service logic."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from homesec.config.loader import ConfigError
from homesec.config.manager import ConfigManager
from homesec.models.config import FastAPIServerConfig
from homesec.models.setup import FinalizeRequest
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
        config_manager: ConfigManager | None = None,
    ) -> None:
        self.bootstrap_mode = bootstrap_mode
        self.pipeline_running = pipeline_running
        self._repository = repository
        self._server_config = server_config
        self.config_manager = config_manager or ConfigManager(Path("config/config.yaml"))
        self.restart_requested = False
        if has_cameras:
            cameras = [SimpleNamespace(name="front")]
        else:
            cameras = []
        self._config = SimpleNamespace(
            cameras=cameras,
            server=server_config,
            storage=SimpleNamespace(paths=SimpleNamespace(clips_dir=clips_dir)),
            state_store=SimpleNamespace(dsn_env="DB_DSN", dsn=None),
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

    def request_restart(self) -> None:
        self.restart_requested = True


def _write_existing_config(path: Path) -> ConfigManager:
    payload = {
        "version": 1,
        "cameras": [
            {
                "name": "existing",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp/existing"}},
            }
        ],
        "storage": {"backend": "local", "config": {"root": "./storage"}},
        "state_store": {"dsn_env": "DB_DSN"},
        "notifiers": [{"backend": "mqtt", "config": {"host": "localhost", "port": 1883}}],
        "filter": {"backend": "yolo", "config": {"classes": ["person"], "min_confidence": 0.5}},
        "vlm": {
            "backend": "openai",
            "run_mode": "never",
            "config": {"api_key_env": "OPENAI_API_KEY", "model": "gpt-4o"},
        },
        "alert_policy": {"backend": "default", "config": {"min_risk_level": "high"}},
        "server": {"enabled": True, "host": "0.0.0.0", "port": 8081},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return ConfigManager(path)


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
    monkeypatch.setenv("DB_DSN", "postgresql://user:pass@localhost/homesec")
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
    assert by_name["config_env"].passed is True


@pytest.mark.asyncio
async def test_run_preflight_probes_postgres_via_dsn_in_bootstrap_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should probe postgres via DSN in bootstrap mode when repository is absent."""
    # Given: A bootstrap app without repository but with DB_DSN configured
    monkeypatch.setenv("DB_DSN", "postgresql://user:pass@localhost/homesec")
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
    monkeypatch.setattr(setup_service, "_probe_postgres_dsn", lambda dsn: asyncio.sleep(0, True))

    # When: Running setup preflight
    response = await setup_service.run_preflight(app)

    # Then: Postgres check passes via bootstrap DSN probe and aggregate result is successful
    by_name = {check.name: check for check in response.checks}
    assert by_name["postgres"].passed is True
    assert by_name["postgres"].message == "Database reachable"
    assert by_name["ffmpeg"].passed is True
    assert by_name["disk_space"].passed is True
    assert by_name["network"].passed is True
    assert by_name["config_env"].passed is True
    assert response.all_passed is True


@pytest.mark.asyncio
async def test_run_preflight_marks_check_failed_when_timeout_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should fail checks that exceed timeout budget."""
    # Given: A preflight run where postgres check exceeds configured timeout budget
    monkeypatch.setenv("DB_DSN", "postgresql://user:pass@localhost/homesec")
    app = _StubApp(
        bootstrap_mode=False,
        pipeline_running=False,
        has_cameras=True,
        repository=_StubRepository(ok=True),
        server_config=_server_config(auth_enabled=False),
    )

    async def _slow_postgres(_: object) -> setup_service.PreflightCheckResponse:
        await asyncio.sleep(0.02)
        return setup_service.PreflightCheckResponse(
            name="postgres",
            passed=True,
            message="Database reachable",
            latency_ms=1.0,
        )

    monkeypatch.setattr(setup_service, "_PREFLIGHT_CHECK_TIMEOUT_S", 0.005)
    monkeypatch.setattr(setup_service, "_postgres_check", _slow_postgres)
    monkeypatch.setattr(setup_service, "shutil_which_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        setup_service,
        "shutil_disk_usage",
        lambda path: (10_000_000_000, 2_000_000_000, 8_000_000_000),
    )
    monkeypatch.setattr(setup_service, "_network_probe", lambda: (True, "DNS resolution succeeded"))

    # When: Running setup preflight
    response = await setup_service.run_preflight(app)

    # Then: Timed-out check is marked failed with timeout message
    by_name = {check.name: check for check in response.checks}
    assert by_name["postgres"].passed is False
    assert "timed out" in by_name["postgres"].message.lower()
    assert response.all_passed is False


@pytest.mark.asyncio
async def test_run_preflight_reports_missing_state_store_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should keep bootstrap config-env readiness advisory and non-blocking."""
    # Given: A bootstrap app with no DB_DSN env value configured
    monkeypatch.delenv("DB_DSN", raising=False)
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

    # Then: config_env check is advisory in bootstrap mode and does not fail the check itself
    by_name = {check.name: check for check in response.checks}
    assert by_name["postgres"].passed is False
    assert "Database DSN not configured" in by_name["postgres"].message
    assert by_name["config_env"].passed is True
    assert "validated at finalize" in by_name["config_env"].message
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


def test_network_probe_target_uses_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Network probe target should honor explicit host/port env overrides."""
    # Given: Environment overrides for setup network probe host and port
    monkeypatch.setenv("HOMESEC_SETUP_NETWORK_PROBE_HOST", "router.local")
    monkeypatch.setenv("HOMESEC_SETUP_NETWORK_PROBE_PORT", "5353")

    # When: Resolving network probe target
    host, port = setup_service._network_probe_target()

    # Then: Target uses configured host and port values
    assert host == "router.local"
    assert port == 5353


def test_network_probe_returns_config_error_for_invalid_port_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Network probe should fail fast when configured port value is invalid."""
    # Given: Invalid setup network probe port environment value
    monkeypatch.setenv("HOMESEC_SETUP_NETWORK_PROBE_PORT", "not-an-int")

    # When: Executing DNS probe with invalid config
    ok, message = setup_service._network_probe()

    # Then: Probe fails with explicit configuration error message
    assert ok is False
    assert "HOMESEC_SETUP_NETWORK_PROBE_PORT must be an integer" in message


@pytest.mark.asyncio
async def test_finalize_setup_writes_config_with_defaults_and_requests_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finalize should persist merged config and request restart in bootstrap mode."""
    # Given: A bootstrap app without existing config and a finalize payload with camera only
    monkeypatch.setenv("DB_DSN", "postgresql://user:pass@localhost/homesec")
    config_path = tmp_path / "config.yaml"
    manager = ConfigManager(config_path)
    app = _StubApp(
        bootstrap_mode=True,
        pipeline_running=False,
        has_cameras=False,
        repository=None,
        server_config=_server_config(auth_enabled=False),
        config_manager=manager,
    )
    request = FinalizeRequest(
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp/front"}},
            }
        ]
    )

    # When: Finalizing setup
    response = await setup_service.finalize_setup(request, app)

    # Then: Config is written, defaults are applied for omitted sections, and restart is requested
    assert response.success is True
    assert response.restart_requested is True
    assert app.restart_requested is True
    assert "storage" in response.defaults_applied
    assert "vlm" in response.defaults_applied
    persisted = manager.get_config()
    assert persisted.cameras[0].name == "front"
    assert persisted.storage.backend == "local"
    assert persisted.state_store.dsn_env == "DB_DSN"
    assert persisted.vlm.run_mode == "never"


@pytest.mark.asyncio
async def test_finalize_setup_validate_only_returns_success_without_writing_or_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate-only finalize should run full validation without mutating runtime/config."""
    # Given: A bootstrap app and valid camera finalize payload in validate-only mode
    monkeypatch.setenv("DB_DSN", "postgresql://user:pass@localhost/homesec")
    config_path = tmp_path / "config.yaml"
    manager = ConfigManager(config_path)
    app = _StubApp(
        bootstrap_mode=True,
        pipeline_running=False,
        has_cameras=False,
        repository=None,
        server_config=_server_config(auth_enabled=False),
        config_manager=manager,
    )
    request = FinalizeRequest(
        validate_only=True,
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp/front"}},
            }
        ],
    )

    # When: Finalizing setup in validation-only mode
    response = await setup_service.finalize_setup(request, app)

    # Then: Service validates and reports success without writing config or requesting restart
    assert response.success is True
    assert response.restart_requested is False
    assert app.restart_requested is False
    assert response.config_path == str(config_path)
    assert "storage" in response.defaults_applied
    assert config_path.exists() is False


@pytest.mark.asyncio
async def test_finalize_setup_returns_validation_error_without_writing_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finalize should surface config validation errors without writing config."""
    # Given: A bootstrap app and an invalid camera source payload
    monkeypatch.setenv("DB_DSN", "postgresql://user:pass@localhost/homesec")
    config_path = tmp_path / "config.yaml"
    manager = ConfigManager(config_path)
    app = _StubApp(
        bootstrap_mode=True,
        pipeline_running=False,
        has_cameras=False,
        repository=None,
        server_config=_server_config(auth_enabled=False),
        config_manager=manager,
    )
    request = FinalizeRequest(
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {
                    "backend": "unknown_source",
                    "config": {},
                },
            }
        ]
    )

    # When/Then: Finalizing setup raises config validation error
    with pytest.raises(ConfigError):
        await setup_service.finalize_setup(request, app)

    # Then: No config is written and restart is not requested
    assert app.restart_requested is False
    assert config_path.exists() is False


@pytest.mark.asyncio
async def test_finalize_setup_reuses_existing_sections_when_payload_omits_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finalize should preserve existing config sections when request leaves them unset."""
    # Given: A bootstrap app with an existing config file and an empty finalize payload
    monkeypatch.setenv("DB_DSN", "postgresql://user:pass@localhost/homesec")
    config_path = tmp_path / "config.yaml"
    manager = _write_existing_config(config_path)
    app = _StubApp(
        bootstrap_mode=True,
        pipeline_running=False,
        has_cameras=False,
        repository=None,
        server_config=_server_config(auth_enabled=False),
        config_manager=manager,
    )
    request = FinalizeRequest()

    # When: Finalizing setup
    response = await setup_service.finalize_setup(request, app)

    # Then: Existing config is preserved and no default sections are reported
    assert response.success is True
    assert response.defaults_applied == []
    persisted = manager.get_config()
    assert persisted.cameras[0].name == "existing"


@pytest.mark.asyncio
async def test_finalize_setup_rejects_empty_camera_set_on_fresh_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalize should reject requests that would persist zero cameras."""
    # Given: A fresh bootstrap app and finalize payload with no camera section
    monkeypatch.setenv("DB_DSN", "postgresql://user:pass@localhost/homesec")
    config_path = tmp_path / "config.yaml"
    manager = ConfigManager(config_path)
    app = _StubApp(
        bootstrap_mode=True,
        pipeline_running=False,
        has_cameras=False,
        repository=None,
        server_config=_server_config(auth_enabled=False),
        config_manager=manager,
    )

    # When/Then: Finalizing setup with empty payload raises semantic validation error
    with pytest.raises(setup_service.SetupFinalizeValidationError) as exc_info:
        await setup_service.finalize_setup(FinalizeRequest(), app)

    # Then: Validation error reports missing-camera requirement and does not write config
    errors = exc_info.value.errors
    assert app.restart_requested is False
    assert any("At least one camera" in error for error in errors)
    assert config_path.exists() is False


@pytest.mark.asyncio
async def test_finalize_setup_rejects_missing_state_store_env_when_defaults_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalize should reject when default state-store dsn_env is unset."""
    # Given: A bootstrap app with camera payload but missing DB_DSN environment variable
    monkeypatch.delenv("DB_DSN", raising=False)
    config_path = tmp_path / "config.yaml"
    manager = ConfigManager(config_path)
    app = _StubApp(
        bootstrap_mode=True,
        pipeline_running=False,
        has_cameras=False,
        repository=None,
        server_config=_server_config(auth_enabled=False),
        config_manager=manager,
    )
    request = FinalizeRequest(
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp/front"}},
            }
        ]
    )

    # When/Then: Finalizing setup raises semantic validation error
    with pytest.raises(setup_service.SetupFinalizeValidationError) as exc_info:
        await setup_service.finalize_setup(request, app)

    # Then: Error includes missing-env guidance and no config is written
    errors = exc_info.value.errors
    assert app.restart_requested is False
    assert any("DB_DSN" in error for error in errors)
    assert config_path.exists() is False
