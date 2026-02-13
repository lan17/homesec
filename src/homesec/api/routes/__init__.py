"""API route registration."""

from __future__ import annotations

from fastapi import Depends, FastAPI

from homesec.api.dependencies import require_database, verify_api_key
from homesec.api.routes import cameras, clips, config, health, notifiers, stats, system


def register_routes(app: FastAPI) -> None:
    """Register all API routers."""
    app.include_router(health.router, dependencies=[Depends(verify_api_key)])
    app.include_router(
        config.router,
        dependencies=[Depends(verify_api_key), Depends(require_database)],
    )
    app.include_router(
        cameras.router,
        dependencies=[Depends(verify_api_key), Depends(require_database)],
    )
    app.include_router(
        clips.router,
        dependencies=[Depends(verify_api_key), Depends(require_database)],
    )
    app.include_router(
        stats.router,
        dependencies=[Depends(verify_api_key), Depends(require_database)],
    )
    app.include_router(
        system.router,
        dependencies=[Depends(verify_api_key), Depends(require_database)],
    )
    app.include_router(notifiers.router, dependencies=[Depends(verify_api_key)])
