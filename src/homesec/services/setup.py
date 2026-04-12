"""Setup/onboarding service logic."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

from pydantic import ValidationError

from homesec.config.loader import ConfigError, ConfigErrorCode
from homesec.models.config import (
    AlertPolicyConfig,
    Config,
    FastAPIServerConfig,
    StateStoreConfig,
    StorageConfig,
)
from homesec.models.enums import VLMRunMode
from homesec.models.filter import FilterConfig
from homesec.models.setup import (
    FinalizeRequest,
    FinalizeResponse,
    PreflightCheckResponse,
    PreflightResponse,
    SetupState,
    SetupStatusResponse,
    TestConnectionRequest,
    TestConnectionResponse,
)
from homesec.models.vlm import VLMConfig
from homesec.plugins.registry import PluginType, get_plugin_names, load_plugin, validate_plugin
from homesec.services.setup_probe_support import (
    build_test_connection_response,
    format_validation_error,
)
from homesec.services.setup_probes import (
    SetupProbeTarget,
    get_setup_probe,
    get_setup_probe_backends,
    get_setup_probe_timeout,
)
from homesec.state.postgres import PostgresStateStore

if TYPE_CHECKING:
    from homesec.app import Application
    from homesec.repository.clip_repository import ClipRepository

logger = logging.getLogger(__name__)

_MIN_FREE_BYTES = 1_000_000_000
_NETWORK_PROBE_HOST_ENV = "HOMESEC_SETUP_NETWORK_PROBE_HOST"
_NETWORK_PROBE_PORT_ENV = "HOMESEC_SETUP_NETWORK_PROBE_PORT"
_NETWORK_PROBE_DEFAULT_HOST = "example.com"
_NETWORK_PROBE_DEFAULT_PORT = 443
_PREFLIGHT_CHECK_TIMEOUT_S = 10.0
_PLUGIN_TEST_CONNECTION_TIMEOUT_S = 5.0
SectionT = TypeVar("SectionT")


class SetupFinalizeValidationError(ValueError):
    """Raised when finalize payload/config is semantically invalid."""

    def __init__(self, errors: list[str]) -> None:
        message = "; ".join(errors) if errors else "Setup finalize validation failed"
        super().__init__(message)
        self.errors = errors


class SetupTestConnectionRequestError(ValueError):
    """Raised when test-connection payload cannot be dispatched."""

    def __init__(
        self,
        message: str,
        *,
        available_backends: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.available_backends = available_backends


class _PingablePlugin(Protocol):
    async def ping(self) -> bool: ...

    async def shutdown(self, timeout: float | None = None) -> None: ...


async def get_setup_status(app: Application) -> SetupStatusResponse:
    """Return setup completion status for onboarding orchestration."""
    config = _active_config(app)
    has_config = config is not None
    has_cameras = bool(config.cameras) if config is not None else False
    pipeline_running = app.pipeline_running
    auth_configured = _auth_configured(_server_config(app))

    state = _resolve_state(
        has_config=has_config,
        has_cameras=has_cameras,
        pipeline_running=pipeline_running,
    )
    return SetupStatusResponse(
        state=state,
        has_cameras=has_cameras,
        pipeline_running=pipeline_running,
        auth_configured=auth_configured,
    )


async def run_preflight(app: Application) -> PreflightResponse:
    """Run setup preflight checks in parallel."""
    checks = await asyncio.gather(
        _run_preflight_check(name="postgres", check=_postgres_check(app)),
        _run_preflight_check(name="ffmpeg", check=_ffmpeg_check()),
        _run_preflight_check(name="disk_space", check=_disk_space_check(app)),
        _run_preflight_check(name="network", check=_network_check()),
        _run_preflight_check(name="config_env", check=_config_env_check(app)),
    )
    return PreflightResponse(
        checks=list(checks),
        all_passed=all(check.passed for check in checks),
    )


async def test_connection(
    request: TestConnectionRequest,
    app: Application,
) -> TestConnectionResponse:
    """Validate and probe connectivity for setup-configured integrations."""
    async with app.setup_test_connection_lock:
        connection_type = request.type
        backend = request.backend.strip().lower()

        match connection_type:
            case "camera":
                return await _test_camera_connection(backend=backend, config=request.config)
            case "storage":
                return await _test_storage_connection(backend=backend, config=request.config)
            case "notifier":
                return await _test_notifier_connection(backend=backend, config=request.config)
            case "analyzer":
                return await _test_analyzer_connection(backend=backend, config=request.config)


async def _test_camera_connection(
    *,
    backend: str,
    config: dict[str, object],
) -> TestConnectionResponse:
    probe_result = await _run_registered_setup_probe("camera", backend, config)
    if probe_result is not None:
        return probe_result

    available_backends = _camera_backend_names()
    if backend not in available_backends:
        raise SetupTestConnectionRequestError(
            (
                f"Unknown camera backend {backend!r}. "
                f"Available backends: {', '.join(available_backends)}"
            ),
            available_backends=available_backends,
        )

    match backend:
        case _:
            # Defensive fallback for known-but-not-yet-probed camera source plugins.
            return build_test_connection_response(
                success=False,
                message=f"Camera backend {backend!r} does not implement connection testing yet.",
            )


async def _test_storage_connection(
    *,
    backend: str,
    config: dict[str, object],
) -> TestConnectionResponse:
    probe_result = await _run_registered_setup_probe("storage", backend, config)
    if probe_result is not None:
        return probe_result
    return await _test_plugin_ping_connection(
        plugin_type=PluginType.STORAGE,
        backend=backend,
        config=config,
    )


async def _test_notifier_connection(
    *,
    backend: str,
    config: dict[str, object],
) -> TestConnectionResponse:
    probe_result = await _run_registered_setup_probe("notifier", backend, config)
    if probe_result is not None:
        return probe_result

    return await _test_plugin_ping_connection(
        plugin_type=PluginType.NOTIFIER,
        backend=backend,
        config=config,
    )


async def _test_analyzer_connection(
    *,
    backend: str,
    config: dict[str, object],
) -> TestConnectionResponse:
    probe_result = await _run_registered_setup_probe("analyzer", backend, config)
    if probe_result is not None:
        return probe_result

    return await _test_plugin_ping_connection(
        plugin_type=PluginType.ANALYZER,
        backend=backend,
        config=config,
    )


async def _test_plugin_ping_connection(
    *,
    plugin_type: PluginType,
    backend: str,
    config: dict[str, object],
) -> TestConnectionResponse:
    start = time.perf_counter()
    _ensure_known_backend(plugin_type, backend)

    try:
        validated_config = validate_plugin(plugin_type, backend, config)
    except ValidationError as exc:
        return build_test_connection_response(
            success=False,
            message=format_validation_error(exc),
            start=start,
        )

    plugin: _PingablePlugin | None = None
    try:
        plugin = cast(
            _PingablePlugin,
            await asyncio.wait_for(
                asyncio.to_thread(load_plugin, plugin_type, backend, validated_config),
                timeout=_PLUGIN_TEST_CONNECTION_TIMEOUT_S,
            ),
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Setup test plugin load timed out for %s backend=%s",
            plugin_type.value,
            backend,
            exc_info=True,
        )
        return build_test_connection_response(
            success=False,
            message=(
                f"{plugin_type.value} probe load timed out after "
                f"{_PLUGIN_TEST_CONNECTION_TIMEOUT_S:.1f}s."
            ),
            start=start,
        )
    except Exception:
        logger.warning(
            "Setup test plugin load failed for %s backend=%s",
            plugin_type.value,
            backend,
            exc_info=True,
        )
        return build_test_connection_response(
            success=False,
            message=(
                f"{plugin_type.value} probe failed during plugin load. "
                "Check backend configuration and connectivity."
            ),
            start=start,
        )

    try:
        ok = await asyncio.wait_for(plugin.ping(), timeout=_PLUGIN_TEST_CONNECTION_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning(
            "Setup test plugin ping timed out for %s backend=%s",
            plugin_type.value,
            backend,
            exc_info=True,
        )
        return build_test_connection_response(
            success=False,
            message=(
                f"{plugin_type.value} probe ping timed out after "
                f"{_PLUGIN_TEST_CONNECTION_TIMEOUT_S:.1f}s."
            ),
            start=start,
        )
    except Exception:
        logger.warning(
            "Setup test plugin ping failed for %s backend=%s",
            plugin_type.value,
            backend,
            exc_info=True,
        )
        return build_test_connection_response(
            success=False,
            message=(
                f"{plugin_type.value} probe failed during ping. "
                "Check backend configuration and connectivity."
            ),
            start=start,
        )
    finally:
        if plugin is not None:
            await _shutdown_plugin(plugin)

    if not ok:
        return build_test_connection_response(
            success=False,
            message=f"{plugin_type.value} probe did not pass health checks.",
            start=start,
        )
    return build_test_connection_response(
        success=True,
        message=f"{plugin_type.value} probe succeeded.",
        start=start,
    )


def _ensure_known_backend(plugin_type: PluginType, backend: str) -> None:
    available = get_plugin_names(plugin_type)
    if backend in available:
        return
    raise SetupTestConnectionRequestError(
        (
            f"Unknown {plugin_type.value} backend {backend!r}. "
            f"Available backends: {', '.join(available)}"
        ),
        available_backends=available,
    )


def _camera_backend_names() -> list[str]:
    available = set(get_plugin_names(PluginType.SOURCE))
    available.update(get_setup_probe_backends("camera"))
    return sorted(available)


async def _run_registered_setup_probe(
    target: SetupProbeTarget,
    backend: str,
    config: dict[str, object],
) -> TestConnectionResponse | None:
    start = time.perf_counter()
    probe = get_setup_probe(target, backend)
    if probe is None:
        return None
    timeout_s = get_setup_probe_timeout(target, backend)
    try:
        if timeout_s is None:
            return await probe(config=config)
        return await asyncio.wait_for(probe(config=config), timeout=timeout_s)
    except SetupTestConnectionRequestError:
        raise
    except ValidationError as exc:
        return build_test_connection_response(
            success=False,
            message=format_validation_error(exc),
            start=start,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Setup registered probe timed out for %s backend=%s",
            target,
            backend,
            exc_info=True,
        )
        return build_test_connection_response(
            success=False,
            message=f"{target} probe timed out after {timeout_s:.1f}s.",
            start=start,
        )
    except Exception:
        logger.warning(
            "Setup registered probe failed for %s backend=%s",
            target,
            backend,
            exc_info=True,
        )
        return build_test_connection_response(
            success=False,
            message=(
                f"{target} probe failed during connectivity test. "
                "Check backend configuration and connectivity."
            ),
            start=start,
        )


async def _shutdown_plugin(plugin: _PingablePlugin) -> None:
    try:
        await asyncio.wait_for(plugin.shutdown(timeout=5.0), timeout=5.0)
    except Exception:
        # Shutdown failure should not hide probe result.
        return


async def _run_preflight_check(
    *,
    name: str,
    check: Awaitable[PreflightCheckResponse],
) -> PreflightCheckResponse:
    """Run a preflight check with timeout protection."""
    start = time.perf_counter()
    try:
        return await asyncio.wait_for(check, timeout=_PREFLIGHT_CHECK_TIMEOUT_S)
    except asyncio.TimeoutError:
        return PreflightCheckResponse(
            name=name,
            passed=False,
            message=(
                f"Check timed out after {_PREFLIGHT_CHECK_TIMEOUT_S:.1f}s. "
                "Verify service availability and network connectivity."
            ),
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return PreflightCheckResponse(
            name=name,
            passed=False,
            message=f"Check failed unexpectedly: {exc}",
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )


async def finalize_setup(request: FinalizeRequest, app: Application) -> FinalizeResponse:
    """Finalize setup with optional validation-only dry run."""
    merged_config, defaults_applied = await _build_finalize_config(request=request, app=app)
    validation_errors = _finalize_validation_errors(merged_config)
    if validation_errors:
        raise SetupFinalizeValidationError(validation_errors)

    config_path = app.config_manager.config_path
    if request.validate_only:
        return FinalizeResponse(
            success=True,
            config_path=str(config_path),
            restart_requested=False,
            defaults_applied=defaults_applied,
            errors=[],
        )

    await app.config_manager.replace_config(merged_config)
    await app.activate_setup_config(merged_config)
    return FinalizeResponse(
        success=True,
        config_path=str(config_path),
        restart_requested=False,
        defaults_applied=defaults_applied,
        errors=[],
    )


def _resolve_state(*, has_config: bool, has_cameras: bool, pipeline_running: bool) -> SetupState:
    if not has_config and not has_cameras and not pipeline_running:
        return "fresh"
    if has_cameras and pipeline_running:
        return "complete"
    return "partial"


async def _build_finalize_config(
    *,
    request: FinalizeRequest,
    app: Application,
) -> tuple[Config, list[str]]:
    existing = await _existing_config_or_none(app)
    defaults_applied: list[str] = []

    cameras = _pick_section(
        requested=request.cameras,
        existing=existing.cameras if existing is not None else None,
        default=[],
        key="cameras",
        defaults_applied=defaults_applied,
    )
    storage = _pick_section(
        requested=request.storage,
        existing=existing.storage if existing is not None else None,
        default=_default_storage(),
        key="storage",
        defaults_applied=defaults_applied,
    )
    state_store = _pick_section(
        requested=request.state_store,
        existing=existing.state_store if existing is not None else None,
        default=_default_state_store(),
        key="state_store",
        defaults_applied=defaults_applied,
    )
    notifiers = _pick_section(
        requested=request.notifiers,
        existing=existing.notifiers if existing is not None else None,
        default=[],
        key="notifiers",
        defaults_applied=defaults_applied,
    )
    filter_config = _pick_section(
        requested=request.filter,
        existing=existing.filter if existing is not None else None,
        default=_default_filter(),
        key="filter",
        defaults_applied=defaults_applied,
    )
    vlm = _pick_section(
        requested=request.vlm,
        existing=existing.vlm if existing is not None else None,
        default=_default_vlm(),
        key="vlm",
        defaults_applied=defaults_applied,
    )
    alert_policy = _pick_section(
        requested=request.alert_policy,
        existing=existing.alert_policy if existing is not None else None,
        default=_default_alert_policy(),
        key="alert_policy",
        defaults_applied=defaults_applied,
    )
    server = _pick_section(
        requested=request.server,
        existing=existing.server if existing is not None else None,
        default=_server_config(app),
        key="server",
        defaults_applied=defaults_applied,
    )

    version = existing.version if existing is not None else 1
    return Config(
        version=version,
        cameras=cameras,
        storage=storage,
        state_store=state_store,
        notifiers=notifiers,
        filter=filter_config,
        vlm=vlm,
        alert_policy=alert_policy,
        server=server,
    ), defaults_applied


async def _existing_config_or_none(app: Application) -> Config | None:
    try:
        return await asyncio.to_thread(app.config_manager.get_config)
    except ConfigError as exc:
        if exc.code == ConfigErrorCode.FILE_NOT_FOUND:
            return None
        raise


def _pick_section(
    *,
    requested: SectionT | None,
    existing: SectionT | None,
    default: SectionT,
    key: str,
    defaults_applied: list[str],
) -> SectionT:
    if requested is not None:
        return requested
    if existing is not None:
        return existing
    defaults_applied.append(key)
    return default


def _default_storage() -> StorageConfig:
    return StorageConfig(backend="local", config={"root": "./storage"})


def _default_state_store() -> StateStoreConfig:
    return StateStoreConfig(dsn_env="DB_DSN")


def _default_filter() -> FilterConfig:
    return FilterConfig(
        backend="yolo",
        config={
            "classes": ["person", "car", "dog", "cat"],
            "min_confidence": 0.5,
        },
    )


def _default_vlm() -> VLMConfig:
    return VLMConfig(
        backend="openai",
        run_mode=VLMRunMode.NEVER,
        trigger_classes=["person"],
        config={"api_key_env": "OPENAI_API_KEY", "model": "gpt-4o"},
    )


def _default_alert_policy() -> AlertPolicyConfig:
    return AlertPolicyConfig(
        backend="default",
        enabled=True,
        config={"min_risk_level": "high"},
    )


def _server_config(app: Application) -> FastAPIServerConfig:
    return app.server_config


def _active_config(app: Application) -> Config | None:
    if app.bootstrap_mode:
        return None
    try:
        return app.config
    except RuntimeError:
        return None


def _auth_configured(server_config: FastAPIServerConfig) -> bool:
    if not server_config.auth_enabled:
        return True
    return bool(server_config.get_api_key())


def _state_store_dsn_env_requirement(config: Config) -> str | None:
    if config.state_store.dsn is not None:
        return None
    return config.state_store.dsn_env


def _finalize_validation_errors(config: Config) -> list[str]:
    errors: list[str] = []
    if not config.cameras:
        errors.append("At least one camera must be configured before finalizing setup.")

    dsn_env = _state_store_dsn_env_requirement(config)
    if dsn_env and not os.environ.get(dsn_env):
        errors.append(
            f"Required environment variable {dsn_env!r} is not set for state_store.dsn_env."
        )
    return errors


async def _postgres_check(app: Application) -> PreflightCheckResponse:
    repository = _repository_or_none(app)
    start = time.perf_counter()
    if repository is None:
        dsn, dsn_env = _state_store_dsn_for_preflight(app)
        if not dsn:
            env_hint = dsn_env or "DB_DSN"
            return PreflightCheckResponse(
                name="postgres",
                passed=False,
                message=f"Database DSN not configured (set {env_hint!r})",
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )

        ok = await _probe_postgres_dsn(dsn)
        return PreflightCheckResponse(
            name="postgres",
            passed=ok,
            message="Database reachable" if ok else "Database unavailable",
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )

    try:
        ok = await repository.ping()
    except Exception as exc:  # pragma: no cover - defensive
        return PreflightCheckResponse(
            name="postgres",
            passed=False,
            message=f"Database probe failed: {exc}",
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
    return PreflightCheckResponse(
        name="postgres",
        passed=ok,
        message="Database reachable" if ok else "Database unavailable",
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )


def _state_store_dsn_for_preflight(app: Application) -> tuple[str | None, str | None]:
    """Resolve DSN/env requirement for bootstrap postgres probing."""
    active_config = _active_config(app)
    if active_config is not None:
        if active_config.state_store.dsn is not None:
            return active_config.state_store.dsn, None
        dsn_env = active_config.state_store.dsn_env
        return (os.environ.get(dsn_env), dsn_env) if dsn_env else (None, None)

    dsn_env = _default_state_store().dsn_env
    return (os.environ.get(dsn_env), dsn_env) if dsn_env else (None, None)


async def _probe_postgres_dsn(dsn: str) -> bool:
    """Probe postgres reachability for setup preflight bootstrap mode."""
    store = PostgresStateStore(dsn)
    try:
        initialized = await store.initialize()
        if not initialized:
            return False
        return await store.ping()
    finally:
        await store.shutdown()


def _repository_or_none(app: Application) -> ClipRepository | None:
    if app.bootstrap_mode:
        return None
    try:
        return app.repository
    except RuntimeError:
        return None


async def _ffmpeg_check() -> PreflightCheckResponse:
    start = time.perf_counter()
    ffmpeg_path = await asyncio.to_thread(shutil_which_ffmpeg)
    latency_ms = (time.perf_counter() - start) * 1000.0
    if ffmpeg_path is None:
        return PreflightCheckResponse(
            name="ffmpeg",
            passed=False,
            message="ffmpeg not found in PATH",
            latency_ms=latency_ms,
        )
    return PreflightCheckResponse(
        name="ffmpeg",
        passed=True,
        message=f"ffmpeg found at {ffmpeg_path}",
        latency_ms=latency_ms,
    )


def shutil_which_ffmpeg() -> str | None:
    from shutil import which

    return which("ffmpeg")


async def _disk_space_check(app: Application) -> PreflightCheckResponse:
    disk_path = _disk_probe_path(app)
    start = time.perf_counter()
    try:
        usage = await asyncio.to_thread(shutil_disk_usage, disk_path)
    except Exception as exc:
        return PreflightCheckResponse(
            name="disk_space",
            passed=False,
            message=f"Disk probe failed for {disk_path}: {exc}",
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )

    free_bytes = usage[2]
    latency_ms = (time.perf_counter() - start) * 1000.0
    if free_bytes < _MIN_FREE_BYTES:
        return PreflightCheckResponse(
            name="disk_space",
            passed=False,
            message=f"Low free space at {disk_path}: {free_bytes} bytes available",
            latency_ms=latency_ms,
        )
    return PreflightCheckResponse(
        name="disk_space",
        passed=True,
        message=f"Sufficient free space at {disk_path}: {free_bytes} bytes available",
        latency_ms=latency_ms,
    )


def _disk_probe_path(app: Application) -> Path:
    config = _active_config(app)
    if config is None:
        return Path.cwd()

    clips_dir = Path(config.storage.paths.clips_dir).expanduser()
    if clips_dir.is_absolute():
        candidate = clips_dir
    else:
        candidate = Path.cwd() / clips_dir

    current = candidate
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def shutil_disk_usage(path: Path) -> tuple[int, int, int]:
    from shutil import disk_usage

    return disk_usage(path)


async def _network_check() -> PreflightCheckResponse:
    start = time.perf_counter()
    ok, message = await asyncio.to_thread(_network_probe)
    return PreflightCheckResponse(
        name="network",
        passed=ok,
        message=message,
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )


async def _config_env_check(app: Application) -> PreflightCheckResponse:
    """Check environment-backed config readiness for state store connectivity."""
    start = time.perf_counter()
    active_config = _active_config(app)
    if active_config is None:
        return PreflightCheckResponse(
            name="config_env",
            passed=True,
            message="No active config loaded; state store env checks are validated at finalize.",
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )

    dsn_env = _state_store_dsn_env_requirement(active_config)

    if not dsn_env:
        return PreflightCheckResponse(
            name="config_env",
            passed=True,
            message="State store DSN configured inline; no environment variable required",
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
    if os.environ.get(dsn_env):
        return PreflightCheckResponse(
            name="config_env",
            passed=True,
            message=f"Required state store environment variable {dsn_env!r} is set",
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
    return PreflightCheckResponse(
        name="config_env",
        passed=False,
        message=f"Required state store environment variable {dsn_env!r} is not set",
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )


def _network_probe() -> tuple[bool, str]:
    try:
        host, port = _network_probe_target()
    except ValueError as exc:
        return False, f"DNS probe configuration invalid: {exc}"

    try:
        socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        return False, f"DNS probe failed for {host}:{port}: {exc}"
    return True, f"DNS resolution succeeded for {host}:{port}"


def _network_probe_target() -> tuple[str, int]:
    host = os.getenv(_NETWORK_PROBE_HOST_ENV, _NETWORK_PROBE_DEFAULT_HOST).strip()
    if not host:
        host = _NETWORK_PROBE_DEFAULT_HOST

    port_value = os.getenv(_NETWORK_PROBE_PORT_ENV)
    if port_value is None or not port_value.strip():
        return host, _NETWORK_PROBE_DEFAULT_PORT

    try:
        port = int(port_value)
    except ValueError:
        raise ValueError(
            f"{_NETWORK_PROBE_PORT_ENV} must be an integer, got {port_value!r}"
        ) from None
    if port < 1 or port > 65535:
        raise ValueError(f"{_NETWORK_PROBE_PORT_ENV} must be in range 1..65535, got {port}")
    return host, port
