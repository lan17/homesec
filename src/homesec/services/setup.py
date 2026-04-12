"""Setup/onboarding service logic."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import tempfile
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

from pydantic import BaseModel, Field, ValidationError

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
from homesec.onvif.client import OnvifCameraClient
from homesec.onvif.discovery import discover_cameras
from homesec.onvif.service import (
    DEFAULT_ONVIF_PORT,
    OnvifProbeError,
    OnvifProbeOptions,
    OnvifProbeTimeoutError,
    OnvifService,
)
from homesec.plugins.registry import PluginType, get_plugin_names, load_plugin, validate_plugin
from homesec.plugins.storage.local import LocalStorageConfig
from homesec.services.setup_probes import (
    SetupProbeTarget,
    get_setup_probe,
    get_setup_probe_backends,
    setup_probe,
)
from homesec.sources.ftp import FtpSourceConfig
from homesec.sources.local_folder import LocalFolderSourceConfig
from homesec.sources.rtsp.core import RTSPSourceConfig
from homesec.sources.rtsp.preflight import PreflightError, RTSPStartupPreflight
from homesec.sources.rtsp.url_derivation import derive_detect_rtsp_url
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
_ONVIF_TEST_CONNECTION_TIMEOUT_CAP_S = 30.0
_RTSP_TEST_CONNECTION_TIMEOUT_S = 10.0
_RTSP_PREFLIGHT_COMMAND_TIMEOUT_CAP_S = 8.0
_FTP_TEST_CONNECTION_TIMEOUT_S = 5.0
_ONVIF_TEST_CONNECTION_TIMEOUT_S = 15.0
_PLUGIN_TEST_CONNECTION_TIMEOUT_S = 5.0
_SETUP_TEST_CAMERA_NAME = "setup-test-camera"
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


class _OnvifTestConnectionConfig(BaseModel):
    host: str = Field(min_length=1)
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)
    port: int = Field(default=DEFAULT_ONVIF_PORT, ge=1, le=65535)
    timeout_s: float = Field(
        default=_ONVIF_TEST_CONNECTION_TIMEOUT_S,
        gt=0.0,
        le=_ONVIF_TEST_CONNECTION_TIMEOUT_CAP_S,
    )


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
        case "rtsp":
            return await _test_rtsp_camera_connection(config=config)
        case "ftp":
            return await _test_ftp_camera_connection(config=config)
        case "local_folder":
            return await _test_local_folder_camera_connection(config=config)
        case "onvif":
            return await _test_onvif_camera_connection(config=config)
        case _:
            # Defensive fallback for known-but-not-yet-probed camera source plugins.
            return _result(
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


@setup_probe("camera", "rtsp")
async def _test_rtsp_camera_connection(
    *,
    config: dict[str, object],
) -> TestConnectionResponse:
    start = time.perf_counter()
    try:
        validated = validate_plugin(
            PluginType.SOURCE,
            "rtsp",
            config,
            camera_name=_SETUP_TEST_CAMERA_NAME,
        )
    except ValidationError as exc:
        return _result(
            success=False,
            message=_format_validation_error(exc),
            start=start,
        )

    if not isinstance(validated, RTSPSourceConfig):
        return _result(
            success=False,
            message=f"Unexpected rtsp config model: {type(validated).__name__}",
            start=start,
        )

    primary_url = _resolve_rtsp_primary_url(validated)
    if not primary_url:
        return _result(
            success=False,
            message="RTSP URL not resolved. Provide rtsp_url or set rtsp_url_env.",
            start=start,
        )
    detect_url = _resolve_rtsp_detect_url(validated, primary_url)

    with tempfile.TemporaryDirectory(prefix="homesec-rtsp-probe-") as temp_dir:
        preflight = RTSPStartupPreflight(
            output_dir=Path(temp_dir),
            rtsp_connect_timeout_s=float(validated.stream.connect_timeout_s),
            rtsp_io_timeout_s=float(validated.stream.io_timeout_s),
            # Keep ffmpeg command execution bounded tighter than full request timeout.
            command_timeout_s=min(
                _RTSP_TEST_CONNECTION_TIMEOUT_S,
                _RTSP_PREFLIGHT_COMMAND_TIMEOUT_CAP_S,
            ),
        )
        try:
            outcome = await asyncio.wait_for(
                asyncio.to_thread(
                    preflight.run,
                    camera_name=validated.camera_name or _SETUP_TEST_CAMERA_NAME,
                    primary_rtsp_url=primary_url,
                    detect_rtsp_url=detect_url,
                ),
                timeout=_RTSP_TEST_CONNECTION_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return _result(
                success=False,
                message=f"RTSP probe timed out after {_RTSP_TEST_CONNECTION_TIMEOUT_S:.1f}s.",
                start=start,
            )
        except Exception:
            logger.warning("RTSP setup test probe failed", exc_info=True)
            return _result(
                success=False,
                message="RTSP probe failed. Check stream URL, credentials, and network connectivity.",
                start=start,
            )

    if isinstance(outcome, PreflightError):
        return _result(
            success=False,
            message=outcome.message,
            start=start,
            details={
                "stage": outcome.stage,
                "camera_key": outcome.camera_key,
            },
        )

    return _result(
        success=True,
        message="RTSP probe succeeded.",
        start=start,
        details={
            "session_mode": outcome.diagnostics.session_mode,
            "selected_recording_profile": outcome.diagnostics.selected_recording_profile,
            "probed_streams": len(outcome.diagnostics.probes),
        },
    )


@setup_probe("camera", "ftp")
async def _test_ftp_camera_connection(
    *,
    config: dict[str, object],
) -> TestConnectionResponse:
    start = time.perf_counter()
    try:
        validated = validate_plugin(
            PluginType.SOURCE,
            "ftp",
            config,
            camera_name=_SETUP_TEST_CAMERA_NAME,
        )
    except ValidationError as exc:
        return _result(
            success=False,
            message=_format_validation_error(exc),
            start=start,
        )

    if not isinstance(validated, FtpSourceConfig):
        return _result(
            success=False,
            message=f"Unexpected ftp config model: {type(validated).__name__}",
            start=start,
        )

    try:
        bound_port = await asyncio.wait_for(
            asyncio.to_thread(_probe_tcp_bind, validated.host, int(validated.port)),
            timeout=_FTP_TEST_CONNECTION_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        return _result(
            success=False,
            message=f"FTP bind probe timed out after {_FTP_TEST_CONNECTION_TIMEOUT_S:.1f}s.",
            start=start,
        )
    except OSError:
        logger.warning("FTP setup test probe failed", exc_info=True)
        return _result(
            success=False,
            message="FTP bind probe failed. Check bind host/port availability and permissions.",
            start=start,
        )

    return _result(
        success=True,
        message="FTP listen address is available.",
        start=start,
        details={"host": validated.host, "bound_port": bound_port},
    )


@setup_probe("camera", "local_folder")
async def _test_local_folder_camera_connection(
    *,
    config: dict[str, object],
) -> TestConnectionResponse:
    start = time.perf_counter()
    try:
        validated = validate_plugin(
            PluginType.SOURCE,
            "local_folder",
            config,
            camera_name=_SETUP_TEST_CAMERA_NAME,
        )
    except ValidationError as exc:
        return _result(
            success=False,
            message=_format_validation_error(exc),
            start=start,
        )

    if not isinstance(validated, LocalFolderSourceConfig):
        return _result(
            success=False,
            message=f"Unexpected local_folder config model: {type(validated).__name__}",
            start=start,
        )

    watch_dir = Path(validated.watch_dir).expanduser()
    if not await asyncio.to_thread(watch_dir.exists):
        return _result(
            success=False,
            message=f"Watch directory does not exist: {watch_dir}",
            start=start,
        )
    if not await asyncio.to_thread(watch_dir.is_dir):
        return _result(
            success=False,
            message=f"Watch path is not a directory: {watch_dir}",
            start=start,
        )
    if not await asyncio.to_thread(os.access, watch_dir, os.R_OK | os.W_OK | os.X_OK):
        return _result(
            success=False,
            message=f"Watch directory is not readable/writable: {watch_dir}",
            start=start,
        )

    resolved_watch_dir = await asyncio.to_thread(watch_dir.resolve)

    return _result(
        success=True,
        message="Local folder path is accessible.",
        start=start,
        details={"watch_dir": str(resolved_watch_dir)},
    )


@setup_probe("camera", "onvif")
async def _test_onvif_camera_connection(
    *,
    config: dict[str, object],
) -> TestConnectionResponse:
    start = time.perf_counter()
    try:
        request = _OnvifTestConnectionConfig.model_validate(config)
    except ValidationError as exc:
        return _result(
            success=False,
            message=_format_validation_error(exc),
            start=start,
        )

    service = OnvifService(
        discover_fn=discover_cameras,
        client_factory=OnvifCameraClient,
    )
    timeout_s = request.timeout_s
    try:
        probe_result = await service.probe(
            OnvifProbeOptions(
                host=request.host,
                username=request.username,
                password=request.password,
                port=int(request.port),
                timeout_s=timeout_s,
            )
        )
    except OnvifProbeTimeoutError:
        return _result(
            success=False,
            message=f"ONVIF probe timed out after {timeout_s:.1f}s.",
            start=start,
        )
    except OnvifProbeError:
        logger.warning("ONVIF setup test probe failed", exc_info=True)
        return _result(
            success=False,
            message="ONVIF probe failed. Check host, credentials, and camera reachability.",
            start=start,
        )

    return _result(
        success=True,
        message="ONVIF probe succeeded.",
        start=start,
        details={
            "profiles": len(probe_result.profiles),
            "streams": len(probe_result.streams),
        },
    )


@setup_probe("storage", "local")
async def _test_local_storage_connection(*, config: dict[str, object]) -> TestConnectionResponse:
    start = time.perf_counter()
    _ensure_known_backend(PluginType.STORAGE, "local")
    try:
        validated = validate_plugin(PluginType.STORAGE, "local", config)
    except ValidationError as exc:
        return _result(
            success=False,
            message=_format_validation_error(exc),
            start=start,
        )

    if not isinstance(validated, LocalStorageConfig):
        return _result(
            success=False,
            message=f"Unexpected local storage config model: {type(validated).__name__}",
            start=start,
        )

    root = Path(validated.root).expanduser()
    if await asyncio.to_thread(root.exists):
        if not await asyncio.to_thread(root.is_dir):
            return _result(
                success=False,
                message=f"Storage root is not a directory: {root}",
                start=start,
            )
        if not await asyncio.to_thread(os.access, root, os.R_OK | os.W_OK | os.X_OK):
            return _result(
                success=False,
                message=f"Storage root is not readable/writable: {root}",
                start=start,
            )
        resolved_root = await asyncio.to_thread(root.resolve)
        return _result(
            success=True,
            message="Local storage root is accessible.",
            start=start,
            details={"root": str(resolved_root)},
        )

    parent = await asyncio.to_thread(_nearest_existing_parent, root)
    if parent is None:
        return _result(
            success=False,
            message=f"No existing parent directory for storage root: {root}",
            start=start,
        )
    if not await asyncio.to_thread(os.access, parent, os.W_OK | os.X_OK):
        return _result(
            success=False,
            message=f"Storage root parent is not writable: {parent}",
            start=start,
        )
    return _result(
        success=True,
        message="Local storage root can be created.",
        start=start,
        details={"root": str(root), "writable_parent": str(parent)},
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
        return _result(
            success=False,
            message=_format_validation_error(exc),
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
        return _result(
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
        return _result(
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
        return _result(
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
        return _result(
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
        return _result(
            success=False,
            message=f"{plugin_type.value} probe did not pass health checks.",
            start=start,
        )
    return _result(
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
    probe = get_setup_probe(target, backend)
    if probe is None:
        return None
    return await probe(config=config)


def _resolve_rtsp_primary_url(config: RTSPSourceConfig) -> str | None:
    if config.rtsp_url_env:
        env_url = os.getenv(config.rtsp_url_env)
        if env_url:
            return env_url
    return config.rtsp_url


def _resolve_rtsp_detect_url(config: RTSPSourceConfig, primary_url: str) -> str:
    if config.detect_rtsp_url_env:
        env_url = os.getenv(config.detect_rtsp_url_env)
        if env_url:
            return env_url
    if config.detect_rtsp_url:
        return config.detect_rtsp_url
    derived = derive_detect_rtsp_url(primary_url)
    if derived is not None:
        return derived.url
    return primary_url


def _probe_tcp_bind(host: str, port: int) -> int:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return int(sock.getsockname()[1])


async def _shutdown_plugin(plugin: _PingablePlugin) -> None:
    try:
        await asyncio.wait_for(plugin.shutdown(timeout=5.0), timeout=5.0)
    except Exception:
        # Shutdown failure should not hide probe result.
        return


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path
    while not current.exists():
        if current == current.parent:
            return None
        current = current.parent
    return current


def _format_validation_error(exc: ValidationError) -> str:
    # Return only the first validation issue to keep setup probe UX concise.
    errors = exc.errors(include_url=False)
    if not errors:
        return "Configuration validation failed."
    first = errors[0]
    loc_parts = [str(part) for part in first.get("loc", []) if str(part)]
    location = ".".join(loc_parts)
    message = str(first.get("msg", "invalid value"))
    if location:
        return f"Configuration validation failed at {location}: {message}"
    return f"Configuration validation failed: {message}"


def _result(
    *,
    success: bool,
    message: str,
    start: float | None = None,
    details: dict[str, object] | None = None,
) -> TestConnectionResponse:
    latency_ms = None
    if start is not None:
        latency_ms = (time.perf_counter() - start) * 1000.0
    return TestConnectionResponse(
        success=success,
        message=message,
        latency_ms=latency_ms,
        details=details,
    )


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
