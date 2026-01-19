"""Shared helpers for threaded and async clip sources."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from threading import Event, Thread

from homesec.interfaces import ClipSource
from homesec.models.clip import Clip

logger = logging.getLogger(__name__)


class ThreadedClipSource(ClipSource, ABC):
    """Base class for clip sources that run in a background thread."""

    def __init__(self) -> None:
        self._callback: Callable[[Clip], None] | None = None
        self._thread: Thread | None = None
        self._stop_event = Event()
        self._last_heartbeat = time.monotonic()
        self._started = False

    def register_callback(self, callback: Callable[[Clip], None]) -> None:
        """Register callback to be invoked when a new clip is ready."""
        self._callback = callback

    async def start(self) -> None:
        """Start producing clips in a background thread."""
        if self._thread is not None:
            logger.warning("%s already started", self.__class__.__name__)
            return

        self._started = True
        self._stop_event.clear()
        self._on_start()
        self._thread = Thread(target=self._run_wrapper, daemon=True)
        self._thread.start()
        self._on_started()

    def stop(self, timeout: float | None = None) -> None:
        """Stop the background thread and cleanup resources."""
        thread = self._thread
        if thread is None:
            return

        self._stop_event.set()
        self._on_stop()

        if thread.is_alive():
            thread.join(timeout=timeout or self._stop_timeout())

        if self._thread is thread:
            self._thread = None
        self._on_stopped()

    async def shutdown(self, timeout: float | None = None) -> None:
        """Async wrapper for stopping the background thread."""
        await asyncio.to_thread(self.stop, timeout)

    def is_healthy(self) -> bool:
        """Default health check: thread is alive (if started)."""
        return self._thread_is_healthy()

    def last_heartbeat(self) -> float:
        """Return monotonic timestamp of last heartbeat update."""
        return self._last_heartbeat

    async def ping(self) -> bool:
        """Health check - verify source is operational.

        Returns True if:
        - Source not started yet (ready to start)
        - Background thread is alive
        """
        return self._thread_is_healthy()

    def _touch_heartbeat(self) -> None:
        self._last_heartbeat = time.monotonic()

    def _thread_is_healthy(self) -> bool:
        if self._thread is None:
            return not self._started
        return self._thread.is_alive()

    def _emit_clip(self, clip: Clip) -> None:
        if not self._callback:
            return
        try:
            self._callback(clip)
        except Exception as exc:
            logger.error(
                "Callback failed for %s: %s",
                clip.clip_id,
                exc,
                exc_info=True,
            )

    def _run_wrapper(self) -> None:
        try:
            self._run()
        except Exception:
            logger.exception("%s stopped unexpectedly", self.__class__.__name__)
        finally:
            self._thread = None

    def _stop_timeout(self) -> float:
        return 5.0

    def _on_start(self) -> None:
        """Hook called before starting the background thread."""

    def _on_started(self) -> None:
        """Hook called after starting the background thread."""

    def _on_stop(self) -> None:
        """Hook called before stopping the background thread."""

    def _on_stopped(self) -> None:
        """Hook called after stopping the background thread."""

    @abstractmethod
    def _run(self) -> None:
        """Thread entrypoint (blocking)."""
        raise NotImplementedError


class AsyncClipSource(ClipSource, ABC):
    """Base class for clip sources that run as async tasks."""

    def __init__(self) -> None:
        self._callback: Callable[[Clip], None] | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._last_heartbeat = time.monotonic()
        self._started = False

    def register_callback(self, callback: Callable[[Clip], None]) -> None:
        """Register callback to be invoked when a new clip is ready."""
        self._callback = callback

    async def start(self) -> None:
        """Start producing clips in a background task."""
        if self._task is not None:
            logger.warning("%s already started", self.__class__.__name__)
            return

        self._started = True
        self._stop_event.clear()
        self._on_start()
        self._task = asyncio.create_task(self._run_wrapper())
        self._on_started()

    async def shutdown(self, timeout: float | None = None) -> None:
        """Stop the background task and cleanup resources."""
        task = self._task
        if task is None:
            return

        self._stop_event.set()
        self._on_stop()

        if not task.done():
            try:
                await asyncio.wait_for(task, timeout=timeout or self._stop_timeout())
            except asyncio.TimeoutError:
                logger.warning("%s shutdown timed out, cancelling task", self.__class__.__name__)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._task is task:
            self._task = None
        self._on_stopped()

    def is_healthy(self) -> bool:
        """Default health check: task is running (if started)."""
        return self._task_is_healthy()

    def last_heartbeat(self) -> float:
        """Return timestamp (monotonic) of last successful operation."""
        return self._last_heartbeat

    async def ping(self) -> bool:
        """Health check - verify source is operational.

        Returns True if:
        - Source not started yet (ready to start)
        - Background task is running
        """
        return self._task_is_healthy()

    def _touch_heartbeat(self) -> None:
        self._last_heartbeat = time.monotonic()

    def _task_is_healthy(self) -> bool:
        if self._task is None:
            return not self._started
        return not self._task.done()

    def _emit_clip(self, clip: Clip) -> None:
        if not self._callback:
            return
        try:
            self._callback(clip)
        except Exception as exc:
            logger.error(
                "Callback failed for %s: %s",
                clip.clip_id,
                exc,
                exc_info=True,
            )

    async def _run_wrapper(self) -> None:
        try:
            await self._run()
        except Exception:
            logger.exception("%s stopped unexpectedly", self.__class__.__name__)
        finally:
            self._task = None

    def _stop_timeout(self) -> float:
        return 5.0

    def _on_start(self) -> None:
        """Hook called before starting the background task."""

    def _on_started(self) -> None:
        """Hook called after starting the background task."""

    def _on_stop(self) -> None:
        """Hook called before stopping the background task."""

    def _on_stopped(self) -> None:
        """Hook called after stopping the background task."""

    @abstractmethod
    async def _run(self) -> None:
        """Async task entrypoint."""
        raise NotImplementedError
