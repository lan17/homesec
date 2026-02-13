"""FastAPI dependency helpers."""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from fastapi import Depends, HTTPException, Request

if TYPE_CHECKING:
    from homesec.app import Application

DB_PING_CACHE_TTL_S = 0.5


class DatabaseUnavailableError(RuntimeError):
    """Raised when database is unavailable for API requests."""


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
        raise HTTPException(status_code=503, detail="Application not initialized")
    return app


async def verify_api_key(request: Request, app: Application = Depends(get_homesec_app)) -> None:
    """Verify API key if authentication is enabled."""
    path = request.url.path
    if path in ("/api/v1/health", "/api/v1/diagnostics"):
        return

    server_config = app.config.server
    if not server_config.auth_enabled:
        return

    api_key = server_config.get_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = auth_header.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token, api_key):
        raise HTTPException(status_code=401, detail="Unauthorized")


async def require_database(app: Application = Depends(get_homesec_app)) -> None:
    """Ensure the database is reachable for data endpoints."""
    cache = _get_database_probe_cache(app)
    now = time.monotonic()

    if now - cache.last_check_monotonic > DB_PING_CACHE_TTL_S:
        async with cache.lock:
            now = time.monotonic()
            if now - cache.last_check_monotonic > DB_PING_CACHE_TTL_S:
                cache.last_ok = await app.repository.ping()
                cache.last_check_monotonic = now

    if not cache.last_ok:
        raise DatabaseUnavailableError("Database unavailable")
