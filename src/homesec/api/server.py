"""FastAPI server wiring for HomeSec."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from homesec.api.errors import register_exception_handlers
from homesec.api.routes import register_routes

if TYPE_CHECKING:
    from homesec.app import Application

logger = logging.getLogger(__name__)


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

    return app


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
            logger.error("API server stopped with error: %s", exc, exc_info=exc)
        self._task = None
        self._server = None
        logger.info("API server stopped")
