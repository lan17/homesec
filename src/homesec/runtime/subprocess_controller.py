"""Runtime controller that supervises runtime as a subprocess."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import socket
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from homesec.runtime.controller import RuntimeController
from homesec.runtime.models import ManagedRuntime, RuntimeCameraStatus, config_signature
from homesec.runtime.subprocess_protocol import WorkerEvent, WorkerEventType

if TYPE_CHECKING:
    from homesec.models.config import Config

logger = logging.getLogger(__name__)


class _WorkerEventProtocol(asyncio.DatagramProtocol):
    """Parses worker datagrams into typed events."""

    def __init__(self, queue: asyncio.Queue[WorkerEvent], *, generation: int) -> None:
        self._queue = queue
        self._generation = generation

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        _ = addr
        try:
            payload = data.decode("utf-8")
            event = WorkerEvent.model_validate_json(payload)
        except Exception as exc:
            logger.error("Dropped invalid runtime worker datagram: %s", exc, exc_info=exc)
            return

        if event.generation != self._generation:
            logger.warning(
                "Dropped worker event with mismatched generation: expected=%d got=%d",
                self._generation,
                event.generation,
            )
            return
        self._queue.put_nowait(event)


@dataclass(slots=True)
class SubprocessRuntimeHandle:
    """Controller-managed runtime handle for a subprocess generation."""

    generation: int
    config: Config
    config_signature: str
    correlation_id: str
    temp_dir: Path = field(repr=False)
    control_socket_path: Path = field(repr=False)
    config_json_path: Path = field(repr=False)
    event_queue: asyncio.Queue[WorkerEvent] = field(default_factory=asyncio.Queue, repr=False)
    transport: asyncio.DatagramTransport | None = field(default=None, repr=False)
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    event_task: asyncio.Task[None] | None = field(default=None, repr=False)
    wait_task: asyncio.Task[None] | None = field(default=None, repr=False)
    worker_pid: int | None = None
    worker_exit_code: int | None = None
    last_heartbeat_at: datetime | None = None
    last_error: str | None = None
    camera_statuses: dict[str, RuntimeCameraStatus] = field(default_factory=dict)

    @property
    def is_running(self) -> bool:
        process = self.process
        return process is not None and process.returncode is None

    def heartbeat_is_fresh(self, *, max_age_s: float) -> bool:
        if not self.is_running:
            return False
        if self.last_heartbeat_at is None:
            return False
        age_s = (datetime.now(timezone.utc) - self.last_heartbeat_at).total_seconds()
        return age_s <= max_age_s


@dataclass(slots=True)
class SubprocessRuntimeController(RuntimeController):
    """Supervises runtime lifecycle through a worker subprocess."""

    startup_timeout_s: float = 15.0
    shutdown_timeout_s: float = 30.0
    kill_timeout_s: float = 5.0
    worker_heartbeat_interval_s: float = 2.0
    heartbeat_stale_s: float = 10.0
    worker_module: str = "homesec.runtime.worker"
    worker_extra_args: tuple[str, ...] = ()
    _active_runtime: SubprocessRuntimeHandle | None = field(default=None, init=False, repr=False)
    _tracked_handles: dict[int, SubprocessRuntimeHandle] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    async def build_candidate(self, config: Config, generation: int) -> ManagedRuntime:
        self._ensure_posix_support()
        temp_dir = Path(tempfile.gettempdir()) / f"hsrt-{uuid.uuid4().hex[:10]}"
        temp_dir.mkdir(mode=0o700, exist_ok=False)

        config_json_path = temp_dir / "runtime_config.json"
        config_json_path.write_text(config.model_dump_json(), encoding="utf-8")

        control_socket_path = temp_dir / "events.sock"
        queue: asyncio.Queue[WorkerEvent] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            lambda: _WorkerEventProtocol(queue, generation=generation),
            local_addr=str(control_socket_path),
            family=socket.AF_UNIX,
        )

        handle = SubprocessRuntimeHandle(
            generation=generation,
            config=config,
            config_signature=config_signature(config),
            correlation_id=uuid.uuid4().hex,
            temp_dir=temp_dir,
            control_socket_path=control_socket_path,
            config_json_path=config_json_path,
            event_queue=queue,
            transport=transport,
        )
        self._track_handle(handle)
        return handle

    async def start_runtime(self, runtime: ManagedRuntime) -> None:
        handle = self._require_handle(runtime)
        previous = self._active_runtime

        if previous is handle and handle.is_running:
            return

        if previous is not None and previous is not handle:
            await self._stop_handle(previous, finalize=False, context="replace-old-runtime")

        try:
            await self._spawn_and_wait_started(handle)
        except Exception as exc:
            if previous is not None and previous is not handle:
                logger.warning(
                    "Candidate runtime failed; attempting rollback to generation=%d",
                    previous.generation,
                )
                try:
                    await self._spawn_and_wait_started(previous)
                except Exception as rollback_exc:
                    previous.last_error = self._sanitize_error(rollback_exc)
                    await self._finalize_handle(previous)
                    self._active_runtime = None
                    raise RuntimeError(
                        "Runtime candidate start failed and rollback failed: "
                        f"{self._sanitize_error(rollback_exc)}"
                    ) from exc
                self._active_runtime = previous
            raise

        if previous is not None and previous is not handle:
            await self._finalize_handle(previous)

        self._active_runtime = handle

    async def shutdown_runtime(self, runtime: ManagedRuntime) -> None:
        handle = self._require_handle(runtime)
        await self._stop_handle(handle, finalize=True, context="shutdown-runtime")
        if self._active_runtime is handle:
            self._active_runtime = None

    async def shutdown_all(self) -> None:
        handles = list(self._tracked_handles.values())
        errors: list[str] = []

        for handle in handles:
            try:
                await self._stop_handle(
                    handle,
                    finalize=True,
                    context="shutdown-all",
                )
            except Exception as exc:
                errors.append(f"generation={handle.generation} error={self._sanitize_error(exc)}")

        self._active_runtime = None
        if errors:
            summary = "; ".join(errors)
            raise RuntimeError(f"Runtime shutdown_all completed with errors: {summary[:512]}")

    @staticmethod
    def _ensure_posix_support() -> None:
        if os.name != "posix":
            raise RuntimeError(
                "Subprocess runtime supervision is supported only on POSIX platforms"
            )

    @staticmethod
    def _require_handle(runtime: ManagedRuntime) -> SubprocessRuntimeHandle:
        if not isinstance(runtime, SubprocessRuntimeHandle):
            raise TypeError(
                "SubprocessRuntimeController requires SubprocessRuntimeHandle, "
                f"got {type(runtime).__name__}"
            )
        return runtime

    async def _spawn_and_wait_started(self, handle: SubprocessRuntimeHandle) -> None:
        self._drain_queue(handle.event_queue)
        handle.last_error = None
        handle.camera_statuses = {}
        handle.last_heartbeat_at = None

        args = [
            sys.executable,
            "-m",
            self.worker_module,
            "--generation",
            str(handle.generation),
            "--config-json-path",
            str(handle.config_json_path),
            "--control-socket-path",
            str(handle.control_socket_path),
            "--correlation-id",
            handle.correlation_id,
            "--heartbeat-interval-s",
            str(self.worker_heartbeat_interval_s),
            *self.worker_extra_args,
        ]
        process = await asyncio.create_subprocess_exec(*args, start_new_session=True)
        handle.process = process
        handle.worker_pid = process.pid
        handle.worker_exit_code = None

        event = await self._wait_for_start_event(handle, process)
        self._apply_event(handle, event)

        handle.event_task = asyncio.create_task(self._drain_worker_events(handle))
        handle.wait_task = asyncio.create_task(self._watch_worker_exit(handle))
        logger.info(
            "Runtime worker started: generation=%d pid=%d socket=%s",
            handle.generation,
            process.pid,
            handle.control_socket_path,
        )

    async def _wait_for_start_event(
        self,
        handle: SubprocessRuntimeHandle,
        process: asyncio.subprocess.Process,
    ) -> WorkerEvent:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.startup_timeout_s

        while True:
            if process.returncode is not None:
                raise RuntimeError(
                    "Runtime worker exited before startup completed "
                    f"(generation={handle.generation} rc={process.returncode})"
                )

            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for runtime worker startup "
                    f"(generation={handle.generation} timeout_s={self.startup_timeout_s:.1f})"
                )

            try:
                event = await asyncio.wait_for(
                    handle.event_queue.get(),
                    timeout=min(remaining, 0.25),
                )
            except asyncio.TimeoutError:
                continue

            if event.event == WorkerEventType.STARTED:
                return event
            if event.event == WorkerEventType.ERROR:
                message = event.message or "runtime worker startup failed"
                raise RuntimeError(message)

    async def _drain_worker_events(self, handle: SubprocessRuntimeHandle) -> None:
        while True:
            event = await handle.event_queue.get()
            self._apply_event(handle, event)

    async def _watch_worker_exit(self, handle: SubprocessRuntimeHandle) -> None:
        process = handle.process
        if process is None:
            return
        return_code = await process.wait()
        handle.worker_exit_code = return_code
        if return_code != 0 and handle.last_error is None:
            handle.last_error = f"runtime worker exited with code {return_code}"
        logger.info(
            "Runtime worker exited: generation=%d pid=%d rc=%d",
            handle.generation,
            process.pid,
            return_code,
        )

    def _apply_event(self, handle: SubprocessRuntimeHandle, event: WorkerEvent) -> None:
        handle.worker_pid = event.pid
        if event.event in {WorkerEventType.STARTED, WorkerEventType.HEARTBEAT}:
            handle.last_heartbeat_at = event.sent_at
            handle.camera_statuses = {
                camera: RuntimeCameraStatus(
                    healthy=status.healthy,
                    last_heartbeat=status.last_heartbeat,
                )
                for camera, status in event.cameras.items()
            }
            return
        if event.event == WorkerEventType.ERROR:
            handle.last_error = event.message or "runtime worker reported an error"

    async def _stop_handle(
        self,
        handle: SubprocessRuntimeHandle,
        *,
        finalize: bool,
        context: str,
    ) -> None:
        process = handle.process
        stop_error: Exception | None = None
        if process is not None and process.returncode is None:
            try:
                self._signal_process_group(process.pid, signal.SIGTERM)
                await asyncio.wait_for(process.wait(), timeout=self.shutdown_timeout_s)
            except asyncio.TimeoutError:
                logger.warning(
                    "Runtime worker did not exit after SIGTERM; forcing kill: generation=%d pid=%d",
                    handle.generation,
                    process.pid,
                )
                try:
                    self._signal_process_group(process.pid, signal.SIGKILL)
                    await asyncio.wait_for(process.wait(), timeout=self.kill_timeout_s)
                except Exception as exc:
                    stop_error = exc
            except Exception as exc:
                stop_error = exc

        if process is not None:
            handle.worker_exit_code = process.returncode
            if process.returncode is not None:
                handle.process = None
        else:
            handle.process = None

        await self._cancel_task(handle.event_task, context=f"{context}-event-task")
        handle.event_task = None
        await self._cancel_task(handle.wait_task, context=f"{context}-wait-task")
        handle.wait_task = None

        if finalize and handle.process is None:
            await self._finalize_handle(handle)
        elif finalize:
            stop_error = stop_error or RuntimeError(
                "Runtime worker process still running after shutdown attempts"
            )

        if stop_error is not None:
            raise RuntimeError(
                f"{context} failed for generation={handle.generation}"
            ) from stop_error

    async def _finalize_handle(self, handle: SubprocessRuntimeHandle) -> None:
        transport = handle.transport
        if transport is not None:
            transport.close()
            handle.transport = None

        if handle.control_socket_path.exists():
            handle.control_socket_path.unlink(missing_ok=True)
        if handle.config_json_path.exists():
            handle.config_json_path.unlink(missing_ok=True)
        shutil.rmtree(handle.temp_dir, ignore_errors=True)
        self._untrack_handle(handle)

    @staticmethod
    def _signal_process_group(pid: int, sig: signal.Signals) -> None:
        try:
            pgid = os.getpgid(pid)
        except OSError:
            return
        try:
            os.killpg(pgid, sig)
        except OSError:
            return

    async def _cancel_task(self, task: asyncio.Task[None] | None, *, context: str) -> None:
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("%s failed: %s", context, exc, exc_info=exc)

    @staticmethod
    def _drain_queue(queue: asyncio.Queue[WorkerEvent]) -> None:
        while not queue.empty():
            queue.get_nowait()

    @staticmethod
    def _sanitize_error(exc: Exception) -> str:
        value = str(exc).strip()
        if not value:
            value = type(exc).__name__
        return value[:512]

    def _track_handle(self, handle: SubprocessRuntimeHandle) -> None:
        self._tracked_handles[id(handle)] = handle

    def _untrack_handle(self, handle: SubprocessRuntimeHandle) -> None:
        self._tracked_handles.pop(id(handle), None)
