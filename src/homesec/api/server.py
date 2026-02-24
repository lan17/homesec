"""FastAPI server wiring for HomeSec."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from homesec.api.errors import register_exception_handlers
from homesec.api.routes import register_routes
from homesec.models.config import FastAPIServerConfig

if TYPE_CHECKING:
    from homesec.app import Application

logger = logging.getLogger(__name__)
_SPA_EXCLUDED_PREFIXES = ("api", "docs", "redoc")
_SPA_EXCLUDED_EXACT = {"openapi.json", "health"}
_SPA_ROOT_STATIC_FILES = {
    "favicon.ico",
    "robots.txt",
    "manifest.webmanifest",
    "site.webmanifest",
}


def create_contract_app() -> FastAPI:
    """Create a FastAPI app containing only the API contract wiring."""
    app = FastAPI(title="HomeSec API", version="1.0.0")
    register_exception_handlers(app)
    register_routes(app)
    return app


def create_app(app_instance: Application) -> FastAPI:
    """Create the FastAPI application."""
    app = create_contract_app()
    app.state.homesec = app_instance

    server_config = app_instance.config.server
    allow_credentials = "*" not in server_config.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_config.cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _configure_ui_serving(app, server_config)

    return app


def _configure_ui_serving(app: FastAPI, server_config: FastAPIServerConfig) -> None:
    """Serve built SPA assets from FastAPI when enabled."""
    if not server_config.serve_ui:
        return

    dist_dir = Path(server_config.ui_dist_dir).expanduser().resolve()
    index_path = dist_dir / "index.html"
    assets_dir = dist_dir / "assets"

    if not index_path.is_file():
        logger.warning(
            "UI serving enabled but index file not found: %s. Skipping SPA static routes.",
            index_path,
        )
        return

    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="ui-assets")
    else:
        logger.warning("UI assets directory not found at %s", assets_dir)

    @app.get("/", include_in_schema=False)
    async def _serve_ui_index() -> FileResponse:
        return FileResponse(index_path)

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _serve_ui_path(full_path: str) -> FileResponse:
        if _is_reserved_spa_path(full_path):
            raise HTTPException(status_code=404, detail="Not Found")

        requested = (dist_dir / full_path).resolve()
        if not requested.is_relative_to(dist_dir):
            raise HTTPException(status_code=404, detail="Not Found")

        # Serve a constrained allowlist of static files outside /assets.
        if _is_allowed_root_static_file(full_path) and requested.is_file():
            return FileResponse(requested)

        # Treat extensionless paths as SPA client routes.
        if _is_spa_route_path(full_path):
            return FileResponse(index_path)

        # Do not expose arbitrary files from dist (defense in depth).
        raise HTTPException(status_code=404, detail="Not Found")


def _is_allowed_root_static_file(path: str) -> bool:
    normalized = path.strip("/")
    if not normalized:
        return False
    return normalized in _SPA_ROOT_STATIC_FILES


def _is_spa_route_path(path: str) -> bool:
    normalized = path.strip("/")
    if not normalized:
        return True
    return "." not in normalized.split("/")[-1]


def _is_reserved_spa_path(path: str) -> bool:
    """Return True when SPA fallback must not capture this path."""
    normalized = path.strip("/")
    if not normalized:
        return False
    if normalized in _SPA_EXCLUDED_EXACT:
        return True
    first_segment = normalized.split("/", maxsplit=1)[0]
    return first_segment in _SPA_EXCLUDED_PREFIXES


class APIServer:
    """Manages the API server lifecycle."""

    def __init__(
        self,
        app: FastAPI,
        host: str,
        port: int,
        *,
        startup_timeout_s: float = 5.0,
    ) -> None:
        self._app = app
        self._host = host
        self._port = port
        self._startup_timeout_s = startup_timeout_s
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the API server in the background."""
        if self._task is not None:
            return

        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            loop="asyncio",
            log_level="info",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        cast(Any, self._server).install_signal_handlers = False
        self._task = asyncio.create_task(self._server.serve())
        try:
            await self._wait_until_started()
        except Exception:
            self._server.should_exit = True
            with suppress(Exception):
                await self._task
            self._task = None
            self._server = None
            raise

        logger.info("API server started: http://%s:%d", self._host, self._port)

    async def _wait_until_started(self) -> None:
        """Wait until uvicorn reports startup success or startup fails."""
        if self._server is None or self._task is None:
            raise RuntimeError("API server task not initialized")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._startup_timeout_s

        while True:
            if self._task.done():
                self._task.result()
                raise RuntimeError("API server exited before startup completed")

            if self._server.started:
                return

            if loop.time() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for API server startup on {self._host}:{self._port}"
                )

            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        """Stop the API server."""
        if self._server is None or self._task is None:
            return
        self._server.should_exit = True
        try:
            await self._task
        except Exception as exc:
            logger.error("API server stopped with error: %s", exc, exc_info=True)
        self._task = None
        self._server = None
        logger.info("API server stopped")
