"""FastAPI dependency helpers."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, cast

from fastapi import Depends, HTTPException, Request

if TYPE_CHECKING:
    from homesec.app import Application


class DatabaseUnavailableError(RuntimeError):
    """Raised when database is unavailable for API requests."""


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
    repository = app.repository
    ok = await repository.ping()
    if not ok:
        raise DatabaseUnavailableError("Database unavailable")
