"""FastAPI server wiring for HomeSec."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from homesec.api.dependencies import DatabaseUnavailableError
from homesec.api.routes import register_routes

if TYPE_CHECKING:
    from homesec.app import Application

logger = logging.getLogger(__name__)


def create_app(app_instance: Application) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="HomeSec API", version="1.0.0")
    app.state.homesec = app_instance

    server_config = app_instance.config.server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(DatabaseUnavailableError)
    async def _db_unavailable_handler(
        request: object, exc: DatabaseUnavailableError
    ) -> JSONResponse:
        _ = request
        _ = exc
        return JSONResponse(
            status_code=503,
            content={"detail": "Database unavailable", "error_code": "DB_UNAVAILABLE"},
        )

    register_routes(app)
    return app


class APIServer:
    """Manages the API server lifecycle."""

    def __init__(self, app: FastAPI, host: str, port: int) -> None:
        self._app = app
        self._host = host
        self._port = port
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
        logger.info("API server started: http://%s:%d", self._host, self._port)

    async def stop(self) -> None:
        """Stop the API server."""
        if self._server is None or self._task is None:
            return
        self._server.should_exit = True
        await self._task
        self._task = None
        self._server = None
        logger.info("API server stopped")
