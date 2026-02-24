"""Setup/onboarding service logic."""

from __future__ import annotations

import asyncio
import socket
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

from homesec.models.config import Config, FastAPIServerConfig
from homesec.models.setup import (
    PreflightCheckResponse,
    PreflightResponse,
    SetupState,
    SetupStatusResponse,
)

if TYPE_CHECKING:
    from homesec.app import Application
    from homesec.repository.clip_repository import ClipRepository

_MIN_FREE_BYTES = 1_000_000_000


async def get_setup_status(app: Application) -> SetupStatusResponse:
    """Return setup completion status for onboarding orchestration."""
    config = _active_config(app)
    has_config = config is not None
    has_cameras = bool(config.cameras) if config is not None else False
    pipeline_running = bool(getattr(app, "pipeline_running", False))
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
        _postgres_check(app),
        _ffmpeg_check(),
        _disk_space_check(app),
        _network_check(),
    )
    return PreflightResponse(
        checks=list(checks),
        all_passed=all(check.passed for check in checks),
    )


def _resolve_state(*, has_config: bool, has_cameras: bool, pipeline_running: bool) -> SetupState:
    if not has_config and not has_cameras and not pipeline_running:
        return "fresh"
    if has_cameras and pipeline_running:
        return "complete"
    return "partial"


def _server_config(app: Application) -> FastAPIServerConfig:
    server_config = cast(FastAPIServerConfig | None, getattr(app, "server_config", None))
    if server_config is not None:
        return server_config
    return app.config.server


def _active_config(app: Application) -> Config | None:
    if bool(getattr(app, "bootstrap_mode", False)):
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
    if bool(getattr(app, "bootstrap_mode", False)):
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
        socket.getaddrinfo("example.com", 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        return False, f"DNS probe failed: {exc}"
    return True, "DNS resolution succeeded"
