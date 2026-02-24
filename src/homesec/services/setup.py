"""Setup/onboarding service logic."""

from __future__ import annotations

import asyncio
import os
import socket
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from homesec.config.loader import ConfigError, ConfigErrorCode
from homesec.models.config import (
    AlertPolicyConfig,
    Config,
    FastAPIServerConfig,
    NotifierConfig,
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
)
from homesec.models.vlm import VLMConfig

if TYPE_CHECKING:
    from homesec.app import Application
    from homesec.repository.clip_repository import ClipRepository

_MIN_FREE_BYTES = 1_000_000_000
_NETWORK_PROBE_HOST_ENV = "HOMESEC_SETUP_NETWORK_PROBE_HOST"
_NETWORK_PROBE_PORT_ENV = "HOMESEC_SETUP_NETWORK_PROBE_PORT"
_NETWORK_PROBE_DEFAULT_HOST = "example.com"
_NETWORK_PROBE_DEFAULT_PORT = 443
_PREFLIGHT_CHECK_TIMEOUT_S = 10.0
SectionT = TypeVar("SectionT")


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
    )
    return PreflightResponse(
        checks=list(checks),
        all_passed=all(check.passed for check in checks),
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
    """Persist finalized setup config and request graceful restart."""
    merged_config, defaults_applied = await _build_finalize_config(request=request, app=app)

    config_path = app.config_manager.config_path
    try:
        await app.config_manager.replace_config(merged_config)
    except ConfigError as exc:
        return FinalizeResponse(
            success=False,
            config_path=str(config_path),
            restart_requested=False,
            defaults_applied=defaults_applied,
            errors=[str(exc)],
        )

    app.request_restart()
    return FinalizeResponse(
        success=True,
        config_path=str(config_path),
        restart_requested=True,
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
        default=[_default_notifier()],
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


def _default_notifier() -> NotifierConfig:
    return NotifierConfig(backend="mqtt", enabled=True, config={"host": "localhost", "port": 1883})


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


async def _postgres_check(app: Application) -> PreflightCheckResponse:
    repository = _repository_or_none(app)
    if repository is None:
        return PreflightCheckResponse(
            name="postgres",
            passed=False,
            message="Database not configured",
            latency_ms=None,
        )

    start = time.perf_counter()
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
