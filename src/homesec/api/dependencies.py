"""FastAPI dependency helpers."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from fastapi import Depends, Query, Request, status

from homesec.api.errors import APIError, APIErrorCode
from homesec.api.media_tokens import MediaTokenError, validate_clip_media_token
from homesec.models.config import FastAPIServerConfig

if TYPE_CHECKING:
    from homesec.app import Application
    from homesec.models.clip import ClipListCursor, ClipListPage, ClipStateData
    from homesec.models.config import Config
    from homesec.models.enums import ClipStatus
    from homesec.runtime.models import RuntimeReloadRequest, RuntimeStatusSnapshot

DB_PING_CACHE_TTL_S = 0.5
logger = logging.getLogger(__name__)


@dataclass
class _DatabaseProbeCache:
    last_check_monotonic: float = 0.0
    last_ok: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class CameraStatusReadable(Protocol):
    """Minimal camera/source health view needed by API routes."""

    def is_healthy(self) -> bool: ...

    def last_heartbeat(self) -> float | None: ...


class AuthDependencyApp(Protocol):
    """App surface required by auth dependencies."""

    @property
    def server_config(self) -> FastAPIServerConfig: ...


class ModeDependencyApp(Protocol):
    """App surface required by mode-gating dependencies."""

    @property
    def bootstrap_mode(self) -> bool: ...


class DatabaseDependencyApp(ModeDependencyApp, Protocol):
    """App surface required by database availability checks."""

    @property
    def repository(self) -> RepositoryPingReadable: ...


class RepositoryPingReadable(Protocol):
    """Repository surface needed for availability checks."""

    async def ping(self) -> bool: ...


class CameraMutationResultReadable(Protocol):
    """Config mutation result surface consumed by camera routes."""

    restart_required: bool


class CameraConfigManagerReadable(Protocol):
    """Config-manager surface consumed by camera routes."""

    def get_config(self) -> Config: ...

    async def add_camera(
        self,
        *,
        name: str,
        enabled: bool,
        source_backend: str,
        source_config: dict[str, object],
    ) -> CameraMutationResultReadable: ...

    async def update_camera(
        self,
        *,
        camera_name: str,
        enabled: bool | None,
        source_backend: str | None,
        source_config: dict[str, object] | None,
    ) -> CameraMutationResultReadable: ...

    async def remove_camera(self, *, camera_name: str) -> CameraMutationResultReadable: ...


class ConfigReader(Protocol):
    """Config-reader surface consumed by read-only routes."""

    def get_config(self) -> Config: ...


class ClipRoutesRepository(Protocol):
    """Repository surface consumed by clip routes."""

    async def list_clips(
        self,
        *,
        camera: str | None = None,
        status: ClipStatus | None = None,
        alerted: bool | None = None,
        detected: bool | None = None,
        risk_level: str | None = None,
        activity_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        cursor: ClipListCursor | None = None,
        limit: int = 50,
    ) -> ClipListPage: ...

    async def get_clip(self, clip_id: str) -> ClipStateData | None: ...

    async def delete_clip(self, clip_id: str) -> ClipStateData: ...


class StatsRoutesRepository(Protocol):
    """Repository surface consumed by stats routes."""

    async def count_clips_since(self, since: datetime) -> int: ...

    async def count_alerts_since(self, since: datetime) -> int: ...


class StoragePingReadable(Protocol):
    """Storage surface needed for diagnostics."""

    async def ping(self) -> bool: ...


class ClipMediaStorage(Protocol):
    """Storage surface consumed by clip/media routes."""

    async def get(self, storage_uri: str, local_path: Path) -> None: ...

    async def delete(self, storage_uri: str) -> None: ...


class CameraRoutesApp(Protocol):
    """App surface consumed by camera CRUD routes."""

    @property
    def config_manager(self) -> CameraConfigManagerReadable: ...

    def get_source(self, camera_name: str) -> CameraStatusReadable | None: ...

    async def request_runtime_reload(self) -> RuntimeReloadRequest: ...


class ClipRoutesApp(Protocol):
    """App surface consumed by clip browsing/media-token routes."""

    @property
    def repository(self) -> ClipRoutesRepository: ...

    @property
    def storage(self) -> ClipMediaStorage: ...

    @property
    def server_config(self) -> FastAPIServerConfig: ...


class ConfigRoutesApp(Protocol):
    """App surface consumed by config routes."""

    @property
    def config_manager(self) -> ConfigReader: ...


class RuntimeRoutesApp(Protocol):
    """App surface consumed by runtime control routes."""

    def get_runtime_status(self) -> RuntimeStatusSnapshot: ...

    async def request_runtime_reload(self) -> RuntimeReloadRequest: ...


class HealthRoutesApp(Protocol):
    """App surface consumed by health and diagnostics routes."""

    @property
    def bootstrap_mode(self) -> bool: ...

    @property
    def pipeline_running(self) -> bool: ...

    @property
    def repository(self) -> RepositoryPingReadable: ...

    @property
    def storage(self) -> StoragePingReadable: ...

    @property
    def config(self) -> Config: ...

    @property
    def sources(self) -> Sequence[CameraStatusReadable]: ...

    def get_source(self, camera_name: str) -> CameraStatusReadable | None: ...

    @property
    def uptime_seconds(self) -> float: ...


class StatsRoutesApp(Protocol):
    """App surface consumed by stats routes."""

    @property
    def repository(self) -> StatsRoutesRepository: ...

    @property
    def config(self) -> Config: ...

    @property
    def sources(self) -> Sequence[CameraStatusReadable]: ...

    @property
    def uptime_seconds(self) -> float: ...


def _get_database_probe_cache(app: object) -> _DatabaseProbeCache:
    cache = cast(_DatabaseProbeCache | None, getattr(app, "_db_probe_cache", None))
    if cache is None:
        cache = _DatabaseProbeCache()
        cast(Any, app)._db_probe_cache = cache
    return cache


async def get_homesec_app(request: Request) -> Application:
    """Get the HomeSec Application instance from request state."""
    app = cast("Application | None", getattr(request.app.state, "homesec", None))
    if app is None:
        raise APIError(
            "Application not initialized",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=APIErrorCode.APP_NOT_INITIALIZED,
        )
    return app


async def get_auth_dependency_app(
    app: Application = Depends(get_homesec_app),
) -> AuthDependencyApp:
    """Return the minimal app surface needed by auth guards."""
    return app


async def get_mode_dependency_app(
    app: Application = Depends(get_homesec_app),
) -> ModeDependencyApp:
    """Return the minimal app surface needed by mode guards."""
    return app


async def get_database_dependency_app(
    app: Application = Depends(get_homesec_app),
) -> DatabaseDependencyApp:
    """Return the minimal app surface needed by DB availability guards."""
    return app


async def get_camera_routes_app(
    app: Application = Depends(get_homesec_app),
) -> CameraRoutesApp:
    """Return the minimal app surface needed by camera routes."""
    return app


async def get_clip_routes_app(
    app: Application = Depends(get_homesec_app),
) -> ClipRoutesApp:
    """Return the minimal app surface needed by clip/media routes."""
    return app


async def get_config_routes_app(
    app: Application = Depends(get_homesec_app),
) -> ConfigRoutesApp:
    """Return the minimal app surface needed by config routes."""
    return app


async def get_runtime_routes_app(
    app: Application = Depends(get_homesec_app),
) -> RuntimeRoutesApp:
    """Return the minimal app surface needed by runtime routes."""
    return app


async def get_health_routes_app(
    app: Application = Depends(get_homesec_app),
) -> HealthRoutesApp:
    """Return the minimal app surface needed by health routes."""
    return app


async def get_stats_routes_app(
    app: Application = Depends(get_homesec_app),
) -> StatsRoutesApp:
    """Return the minimal app surface needed by stats routes."""
    return app


async def verify_api_key(
    request: Request,
    app: AuthDependencyApp = Depends(get_auth_dependency_app),
) -> None:
    """Verify API key if authentication is enabled."""
    server_config = _get_server_config(app)
    if not server_config.auth_enabled:
        return

    api_key = _require_api_key_value(app)

    token = _parse_bearer_token(request)
    if token is None:
        raise APIError(
            "Unauthorized",
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code=APIErrorCode.UNAUTHORIZED,
        )

    if not secrets.compare_digest(token, api_key):
        raise APIError(
            "Unauthorized",
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code=APIErrorCode.UNAUTHORIZED,
        )


async def verify_media_access(
    request: Request,
    clip_id: str,
    token: str | None = Query(default=None),
    app: AuthDependencyApp = Depends(get_auth_dependency_app),
) -> None:
    """Authorize media playback requests with API key or short-lived media token."""
    server_config = _get_server_config(app)
    if not server_config.auth_enabled:
        return

    api_key = _require_api_key_value(app)
    bearer_token = _parse_bearer_token(request)
    if bearer_token is not None and secrets.compare_digest(bearer_token, api_key):
        return

    if token is None:
        raise APIError(
            "Media token rejected",
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code=APIErrorCode.MEDIA_TOKEN_REJECTED,
        )

    try:
        validate_clip_media_token(
            api_key=api_key,
            token=token,
            clip_id=clip_id,
        )
    except MediaTokenError as exc:
        logger.info(
            "Rejected media token for clip_id=%s reason=%s",
            clip_id,
            exc.code.value,
        )
        raise APIError(
            "Media token rejected",
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code=APIErrorCode.MEDIA_TOKEN_REJECTED,
        ) from exc


async def require_database(
    app: DatabaseDependencyApp = Depends(get_database_dependency_app),
) -> None:
    """Ensure the database is reachable for data endpoints."""
    # Defense in depth: most DB-backed routes are also guarded by require_normal_mode,
    # but this keeps DB-only routes safe if they are ever added without that dependency.
    if _is_bootstrap_mode(app):
        raise APIError(
            "System is in setup mode. Complete the setup wizard first.",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=APIErrorCode.SETUP_REQUIRED,
        )

    cache = _get_database_probe_cache(app)
    now = time.monotonic()

    if now - cache.last_check_monotonic > DB_PING_CACHE_TTL_S:
        async with cache.lock:
            now = time.monotonic()
            if now - cache.last_check_monotonic > DB_PING_CACHE_TTL_S:
                cache.last_ok = await app.repository.ping()
                cache.last_check_monotonic = now

    if not cache.last_ok:
        raise APIError(
            "Database unavailable",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=APIErrorCode.DB_UNAVAILABLE,
        )


async def require_normal_mode(app: ModeDependencyApp = Depends(get_mode_dependency_app)) -> None:
    """Ensure runtime-dependent routes are unavailable in bootstrap mode."""
    if not _is_bootstrap_mode(app):
        return
    raise APIError(
        "System is in setup mode. Complete the setup wizard first.",
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        error_code=APIErrorCode.SETUP_REQUIRED,
    )


async def require_bootstrap_mode(app: ModeDependencyApp = Depends(get_mode_dependency_app)) -> None:
    """Ensure setup-finalize route is only available during bootstrap mode."""
    if _is_bootstrap_mode(app):
        return
    raise APIError(
        "Setup finalize is only available while running in setup mode.",
        status_code=status.HTTP_409_CONFLICT,
        error_code=APIErrorCode.CONFLICT,
    )


def _get_server_config(app: AuthDependencyApp) -> FastAPIServerConfig:
    return app.server_config


def _is_bootstrap_mode(app: ModeDependencyApp) -> bool:
    return bool(app.bootstrap_mode)


def _require_api_key_value(app: AuthDependencyApp) -> str:
    api_key = _get_server_config(app).get_api_key()
    if not api_key:
        raise APIError(
            "API key not configured",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=APIErrorCode.API_KEY_NOT_CONFIGURED,
        )
    return api_key


def _parse_bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    return auth_header.removeprefix("Bearer ").strip()
