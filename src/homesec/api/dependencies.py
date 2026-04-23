"""FastAPI dependency helpers."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from fastapi import Depends, Query, Request, status

from homesec.api.errors import APIError, APIErrorCode
from homesec.api.media_tokens import MediaTokenError, validate_clip_media_token
from homesec.api.preview_tokens import PreviewTokenError, validate_camera_preview_token
from homesec.models.config import FastAPIServerConfig

if TYPE_CHECKING:
    from homesec.app import Application

DB_PING_CACHE_TTL_S = 0.5
logger = logging.getLogger(__name__)


@dataclass
class _DatabaseProbeCache:
    last_check_monotonic: float = 0.0
    last_ok: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _get_database_probe_cache(app: Application) -> _DatabaseProbeCache:
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


async def verify_api_key(request: Request, app: Application = Depends(get_homesec_app)) -> None:
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
    app: Application = Depends(get_homesec_app),
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


async def verify_preview_access(
    request: Request,
    camera_name: str,
    token: str | None = Query(default=None),
    app: Application = Depends(get_homesec_app),
) -> str | None:
    """Authorize preview playback requests with API key or short-lived preview token."""
    server_config = _get_server_config(app)
    if not server_config.auth_enabled:
        return None

    api_key = _require_api_key_value(app)
    bearer_token = _parse_bearer_token(request)
    if bearer_token is not None and secrets.compare_digest(bearer_token, api_key):
        return None

    if token is None:
        raise APIError(
            "Preview token rejected",
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code=APIErrorCode.PREVIEW_TOKEN_REJECTED,
        )

    try:
        validate_camera_preview_token(
            api_key=api_key,
            token=token,
            camera_name=camera_name,
        )
    except PreviewTokenError as exc:
        logger.info(
            "Rejected preview token for camera_name=%s reason=%s",
            camera_name,
            exc.code.value,
        )
        raise APIError(
            "Preview token rejected",
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code=APIErrorCode.PREVIEW_TOKEN_REJECTED,
        ) from exc

    return token


async def require_database(app: Application = Depends(get_homesec_app)) -> None:
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


async def require_normal_mode(app: Application = Depends(get_homesec_app)) -> None:
    """Ensure runtime-dependent routes are unavailable in bootstrap mode."""
    if not _is_bootstrap_mode(app):
        return
    raise APIError(
        "System is in setup mode. Complete the setup wizard first.",
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        error_code=APIErrorCode.SETUP_REQUIRED,
    )


async def require_bootstrap_mode(app: Application = Depends(get_homesec_app)) -> None:
    """Ensure setup-finalize route is only available during bootstrap mode."""
    if _is_bootstrap_mode(app):
        return
    raise APIError(
        "Setup finalize is only available while running in setup mode.",
        status_code=status.HTTP_409_CONFLICT,
        error_code=APIErrorCode.CONFLICT,
    )


def _get_server_config(app: Application) -> FastAPIServerConfig:
    return app.server_config


def _is_bootstrap_mode(app: Application) -> bool:
    return bool(app.bootstrap_mode)


def _require_api_key_value(app: Application) -> str:
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
