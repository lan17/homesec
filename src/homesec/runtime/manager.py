"""Runtime manager for in-process runtime reload orchestration."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from homesec.runtime.controller import RuntimeController
from homesec.runtime.models import (
    ManagedRuntime,
    RuntimeReloadRequest,
    RuntimeReloadResult,
    RuntimeState,
    RuntimeStatusSnapshot,
)

if TYPE_CHECKING:
    from homesec.models.config import Config

logger = logging.getLogger(__name__)


class RuntimeManager:
    """Coordinates runtime lifecycle, reload, and status."""

    def __init__(
        self,
        controller: RuntimeController,
        *,
        on_runtime_activated: Callable[[ManagedRuntime], None] | None = None,
        on_runtime_cleared: Callable[[], None] | None = None,
        reload_shutdown_grace_s: float = 30.0,
        reload_cancel_wait_s: float = 5.0,
    ) -> None:
        self._controller = controller
        self._on_runtime_activated = on_runtime_activated
        self._on_runtime_cleared = on_runtime_cleared
        self._reload_shutdown_grace_s = reload_shutdown_grace_s
        self._reload_cancel_wait_s = reload_cancel_wait_s

        self._active_runtime: ManagedRuntime | None = None
        self._generation = 0
        self._reload_task: asyncio.Task[RuntimeReloadResult] | None = None

        self._status = RuntimeStatusSnapshot(
            state=RuntimeState.IDLE,
            generation=0,
            reload_in_progress=False,
            active_config_version=None,
            last_reload_at=None,
            last_reload_error=None,
        )

    @property
    def active_runtime(self) -> ManagedRuntime | None:
        """Return the currently active runtime bundle, if any."""
        return self._active_runtime

    @property
    def generation(self) -> int:
        """Return current runtime generation."""
        return self._generation

    async def start_initial_runtime(self, config: Config) -> ManagedRuntime:
        """Build and start the initial runtime."""
        if self._active_runtime is not None:
            logger.warning(
                "Initial runtime already active; returning generation=%d",
                self._generation,
            )
            return self._active_runtime

        candidate = await self._controller.build_candidate(config, generation=1)

        try:
            await self._controller.start_runtime(candidate)
        except Exception as exc:
            await self._safe_shutdown(candidate, context="initial runtime cleanup")
            self._status.state = RuntimeState.FAILED
            self._status.last_reload_error = self._sanitize_error(exc)
            self._status.last_reload_at = self._now_utc()
            raise

        self._active_runtime = candidate
        self._generation = candidate.generation
        self._status.state = RuntimeState.IDLE
        self._status.generation = candidate.generation
        self._status.active_config_version = candidate.config_signature

        if self._on_runtime_activated is not None:
            try:
                self._on_runtime_activated(candidate)
            except Exception as exc:
                await self._safe_shutdown(candidate, context="initial runtime activation cleanup")
                error = self._sanitize_error(exc)
                self._rollback_to_previous_runtime(
                    old_runtime=None,
                    old_generation=0,
                    old_config_version=None,
                    error=error,
                )
                logger.error("Initial runtime activation failed: %s", error, exc_info=exc)
                raise RuntimeError(f"Initial runtime activation failed: {error}") from exc

        return candidate

    def request_reload(self, config: Config) -> RuntimeReloadRequest:
        """Start an async runtime reload if one is not already running."""
        target_generation = self._generation + 1 if self._active_runtime is not None else 1

        if self._reload_task is not None and not self._reload_task.done():
            return RuntimeReloadRequest(
                accepted=False,
                message="Runtime reload already in progress",
                target_generation=target_generation,
            )

        self._reload_task = asyncio.create_task(
            self._run_reload(config=config, target_generation=target_generation)
        )
        return RuntimeReloadRequest(
            accepted=True,
            message="Runtime reload started",
            target_generation=target_generation,
        )

    async def wait_for_reload(self) -> RuntimeReloadResult | None:
        """Wait for current reload task completion (if any)."""
        if self._reload_task is None:
            return None
        return await self._reload_task

    def get_status(self) -> RuntimeStatusSnapshot:
        """Return a copy of current runtime status."""
        status = replace(self._status)
        status.reload_in_progress = self._reload_task is not None and not self._reload_task.done()
        return status

    async def shutdown(self) -> None:
        """Shutdown reload task (if any) and active runtime."""
        if self._reload_task is not None and not self._reload_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._reload_task),
                    timeout=self._reload_shutdown_grace_s,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "In-flight reload exceeded shutdown grace period (%.1fs); cancelling task",
                    self._reload_shutdown_grace_s,
                )
                await self._safe_shutdown_all(
                    context="controller shutdown sweep before reload cancel"
                )
                self._reload_task.cancel()
                try:
                    await asyncio.wait_for(
                        self._reload_task,
                        timeout=self._reload_cancel_wait_s,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "Cancelled reload task did not finish within %.1fs",
                        self._reload_cancel_wait_s,
                    )
                except asyncio.CancelledError:
                    pass
            except Exception as exc:
                logger.error("In-flight reload task failed during shutdown: %s", exc, exc_info=exc)
            finally:
                await self._safe_shutdown_all(
                    context="controller shutdown sweep after reload handling"
                )

        active = self._active_runtime
        self._active_runtime = None
        self._generation = 0

        if active is not None:
            await self._safe_shutdown(active, context="active runtime shutdown")

        await self._safe_shutdown_all(context="controller final shutdown sweep")

        self._status = RuntimeStatusSnapshot(
            state=RuntimeState.IDLE,
            generation=0,
            reload_in_progress=False,
            active_config_version=None,
            last_reload_at=self._status.last_reload_at,
            last_reload_error=self._status.last_reload_error,
        )

        if self._on_runtime_cleared is not None:
            self._on_runtime_cleared()

    async def _run_reload(self, config: Config, target_generation: int) -> RuntimeReloadResult:
        self._status.state = RuntimeState.RELOADING

        old_runtime = self._active_runtime
        old_generation = self._generation
        old_config_version = self._status.active_config_version
        candidate: ManagedRuntime | None = None

        try:
            candidate = await self._controller.build_candidate(config, generation=target_generation)
            await self._controller.start_runtime(candidate)
        except Exception as exc:
            if candidate is not None:
                await self._safe_shutdown(candidate, context="failed candidate runtime cleanup")
            error = self._sanitize_error(exc)
            self._rollback_to_previous_runtime(
                old_runtime=old_runtime,
                old_generation=old_generation,
                old_config_version=old_config_version,
                error=error,
            )
            logger.error("Runtime reload failed: %s", error, exc_info=exc)
            return RuntimeReloadResult(success=False, generation=self._generation, error=error)

        self._active_runtime = candidate
        self._generation = target_generation
        self._status.state = RuntimeState.IDLE
        self._status.generation = target_generation
        self._status.active_config_version = candidate.config_signature
        self._status.last_reload_error = None
        self._status.last_reload_at = self._now_utc()

        assert candidate is not None
        if self._on_runtime_activated is not None:
            try:
                self._on_runtime_activated(candidate)
            except Exception as exc:
                error = self._sanitize_error(exc)
                self._rollback_to_previous_runtime(
                    old_runtime=old_runtime,
                    old_generation=old_generation,
                    old_config_version=old_config_version,
                    error=error,
                )
                await self._safe_shutdown(candidate, context="failed candidate runtime cleanup")
                logger.error("Runtime activation callback failed: %s", error, exc_info=exc)
                return RuntimeReloadResult(success=False, generation=self._generation, error=error)

        if old_runtime is not None:
            await self._safe_shutdown(old_runtime, context="previous runtime shutdown")

        return RuntimeReloadResult(success=True, generation=target_generation)

    async def _safe_shutdown(self, runtime: ManagedRuntime, *, context: str) -> None:
        try:
            await self._controller.shutdown_runtime(runtime)
        except Exception as exc:
            logger.error("%s failed: %s", context, exc, exc_info=exc)

    async def _safe_shutdown_all(self, *, context: str) -> None:
        try:
            await self._controller.shutdown_all()
        except Exception as exc:
            logger.error("%s failed: %s", context, exc, exc_info=exc)

    def _rollback_to_previous_runtime(
        self,
        *,
        old_runtime: ManagedRuntime | None,
        old_generation: int,
        old_config_version: str | None,
        error: str,
    ) -> None:
        self._active_runtime = old_runtime
        self._generation = old_generation
        self._status.state = RuntimeState.IDLE if old_runtime is not None else RuntimeState.FAILED
        self._status.generation = old_generation
        self._status.active_config_version = old_config_version
        self._status.last_reload_error = error
        self._status.last_reload_at = self._now_utc()

    @staticmethod
    def _sanitize_error(exc: Exception) -> str:
        value = str(exc).strip()
        if not value:
            value = type(exc).__name__
        return value[:512]

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)
