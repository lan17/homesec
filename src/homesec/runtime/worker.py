"""Runtime worker process entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import socket
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from homesec.config import resolve_env_var
from homesec.logging_setup import configure_logging
from homesec.notifiers.multiplex import MultiplexNotifier, NotifierEntry
from homesec.plugins import discover_all_plugins
from homesec.plugins.alert_policies import load_alert_policy
from homesec.plugins.notifiers import load_notifier_plugin
from homesec.plugins.sources import load_source_plugin
from homesec.plugins.storage import load_storage_plugin
from homesec.repository import ClipRepository
from homesec.runtime.assembly import RuntimeAssembler
from homesec.runtime.models import RuntimeBundle
from homesec.runtime.subprocess_protocol import (
    WorkerCameraStatusPayload,
    WorkerEvent,
    WorkerEventType,
)
from homesec.state import NoopEventStore, PostgresStateStore

if TYPE_CHECKING:
    from homesec.interfaces import (
        AlertPolicy,
        ClipSource,
        EventStore,
        Notifier,
        StateStore,
        StorageBackend,
    )
    from homesec.models.config import Config

logger = logging.getLogger(__name__)


class _WorkerEventEmitter:
    """Sends typed worker events over unix datagram IPC."""

    def __init__(self, socket_path: Path) -> None:
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._socket.connect(str(socket_path))

    def send(self, event: WorkerEvent) -> None:
        payload = event.model_dump_json().encode("utf-8")
        self._socket.send(payload)

    def close(self) -> None:
        self._socket.close()


class _RuntimeWorkerService:
    """Owns runtime worker lifecycle inside subprocess."""

    def __init__(
        self,
        *,
        config: Config,
        generation: int,
        correlation_id: str,
        heartbeat_interval_s: float,
        emitter: _WorkerEventEmitter,
    ) -> None:
        self._config = config
        self._generation = generation
        self._correlation_id = correlation_id
        self._heartbeat_interval_s = heartbeat_interval_s
        self._emitter = emitter

        self._storage: StorageBackend | None = None
        self._state_store: StateStore | None = None
        self._event_store: EventStore | None = None
        self._repository: ClipRepository | None = None
        self._assembler: RuntimeAssembler | None = None
        self._runtime_bundle: RuntimeBundle | None = None

    async def run_runtime(self, stop_event: asyncio.Event) -> None:
        started = False
        heartbeat_task: asyncio.Task[None] | None = None

        try:
            discover_all_plugins()

            storage = load_storage_plugin(self._config.storage)
            self._storage = storage
            self._state_store = await self._create_state_store(self._config)
            self._event_store = self._create_event_store(self._state_store)
            self._repository = ClipRepository(
                self._state_store,
                self._event_store,
                retry=self._config.retry,
            )
            self._assembler = RuntimeAssembler(
                storage=storage,
                repository=self._repository,
                notifier_factory=self._create_notifier,
                notifier_health_logger=self._log_notifier_health,
                alert_policy_factory=self._create_alert_policy,
                source_factory=self._create_sources,
            )

            self._runtime_bundle = await self._assembler.build_bundle(
                self._config,
                generation=self._generation,
            )
            await self._assembler.start_bundle(self._runtime_bundle)
            self._emit_event(WorkerEventType.STARTED)
            started = True

            heartbeat_task = asyncio.create_task(self._heartbeat_loop(stop_event))
            await stop_event.wait()
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

            await self._shutdown_runtime_stack()

            if started:
                self._emit_event(WorkerEventType.STOPPED)

    async def run_test_mode(
        self,
        stop_event: asyncio.Event,
        *,
        fail_startup: bool,
    ) -> None:
        if fail_startup:
            raise RuntimeError("runtime worker test-mode startup failure")

        self._emit_event(WorkerEventType.STARTED)
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(stop_event))
        try:
            await stop_event.wait()
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            self._emit_event(WorkerEventType.STOPPED)

    def emit_error(self, exc: Exception) -> None:
        self._emit_event(
            WorkerEventType.ERROR,
            message=self._sanitize_error(exc),
        )

    async def _heartbeat_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self._heartbeat_interval_s,
                )
            except asyncio.TimeoutError:
                self._emit_event(WorkerEventType.HEARTBEAT)

    async def _shutdown_runtime_stack(self) -> None:
        runtime_bundle = self._runtime_bundle
        if runtime_bundle is not None and self._assembler is not None:
            await self._assembler.shutdown_bundle(runtime_bundle)
            self._runtime_bundle = None

        if self._state_store is not None:
            await self._state_store.shutdown()
            self._state_store = None

        if self._storage is not None:
            await self._storage.shutdown()
            self._storage = None

    async def _create_state_store(self, config: Config) -> StateStore:
        state_cfg = config.state_store
        dsn = state_cfg.dsn
        if state_cfg.dsn_env:
            dsn = resolve_env_var(state_cfg.dsn_env)
        if not dsn:
            raise RuntimeError("Postgres DSN is required for runtime worker state_store backend")
        store = PostgresStateStore(dsn)
        await store.initialize()
        return store

    def _create_event_store(self, state_store: StateStore) -> EventStore:
        event_store = state_store.create_event_store()
        if isinstance(event_store, NoopEventStore):
            logger.warning(
                "Runtime worker event store unavailable (NoopEventStore returned); events dropped"
            )
        return event_store

    def _create_notifier(self, config: Config) -> tuple[Notifier, list[NotifierEntry]]:
        entries: list[NotifierEntry] = []
        for index, notifier_cfg in enumerate(config.notifiers):
            if not notifier_cfg.enabled:
                continue
            notifier = load_notifier_plugin(notifier_cfg.backend, notifier_cfg.config)
            entries.append(
                NotifierEntry(name=f"{notifier_cfg.backend}[{index}]", notifier=notifier)
            )

        if not entries:
            raise RuntimeError("No enabled notifiers configured")
        if len(entries) == 1:
            return entries[0].notifier, entries
        return MultiplexNotifier(entries), entries

    async def _log_notifier_health(self, entries: list[NotifierEntry]) -> None:
        if not entries:
            return
        results = await asyncio.gather(
            *(asyncio.create_task(entry.notifier.ping()) for entry in entries),
            return_exceptions=True,
        )
        for entry, result in zip(entries, results, strict=True):
            match result:
                case bool() as ok:
                    if ok:
                        logger.info("Runtime worker notifier reachable at startup: %s", entry.name)
                    else:
                        logger.error(
                            "Runtime worker notifier unreachable at startup: %s", entry.name
                        )
                case BaseException() as err:
                    logger.error(
                        "Runtime worker notifier ping failed: %s error=%s",
                        entry.name,
                        err,
                        exc_info=err,
                    )

    def _create_alert_policy(self, config: Config) -> AlertPolicy:
        return load_alert_policy(
            config.alert_policy,
            trigger_classes=config.vlm.trigger_classes,
        )

    def _create_sources(self, config: Config) -> tuple[list[ClipSource], dict[str, ClipSource]]:
        sources: list[ClipSource] = []
        sources_by_camera: dict[str, ClipSource] = {}
        runtime_bundle = self._runtime_bundle
        _ = runtime_bundle

        for camera in config.cameras:
            if not camera.enabled:
                continue
            source_cfg = camera.source
            source = load_source_plugin(
                source_backend=source_cfg.backend,
                config=source_cfg.config,
                camera_name=camera.name,
            )
            sources.append(source)
            sources_by_camera[camera.name] = source

        return sources, sources_by_camera

    def _emit_event(self, event_type: WorkerEventType, *, message: str | None = None) -> None:
        event = WorkerEvent(
            event=event_type,
            generation=self._generation,
            correlation_id=self._correlation_id,
            pid=os.getpid(),
            cameras=self._collect_camera_statuses(),
            message=message,
        )
        self._emitter.send(event)

    def _collect_camera_statuses(self) -> dict[str, WorkerCameraStatusPayload]:
        bundle = self._runtime_bundle
        statuses: dict[str, WorkerCameraStatusPayload] = {}
        source_map = bundle.sources_by_camera if bundle is not None else {}

        for camera in self._config.cameras:
            source = source_map.get(camera.name)
            if bundle is None:
                statuses[camera.name] = WorkerCameraStatusPayload(
                    healthy=camera.enabled,
                    last_heartbeat=0.0 if camera.enabled else None,
                )
            elif camera.enabled and source is not None:
                statuses[camera.name] = WorkerCameraStatusPayload(
                    healthy=source.is_healthy(),
                    last_heartbeat=source.last_heartbeat(),
                )
            else:
                statuses[camera.name] = WorkerCameraStatusPayload(
                    healthy=False,
                    last_heartbeat=None,
                )

        return statuses

    @staticmethod
    def _sanitize_error(exc: Exception) -> str:
        value = str(exc).strip()
        if not value:
            value = type(exc).__name__
        return value[:512]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HomeSec runtime worker")
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--config-json-path", type=Path, required=True)
    parser.add_argument("--control-socket-path", type=Path, required=True)
    parser.add_argument("--correlation-id", type=str, required=True)
    parser.add_argument("--heartbeat-interval-s", type=float, default=2.0)
    # TODO(ticket-28 / 306d8336c59f81cba320f7edd58c0cd2): move test-only
    # flags into a dedicated test worker harness once runtime hardening is complete.
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--test-fail-startup", action="store_true")
    parser.add_argument("--test-ignore-term", action="store_true")
    return parser.parse_args(argv)


async def _run_worker(args: argparse.Namespace) -> None:
    config_json = args.config_json_path.read_text(encoding="utf-8")
    from homesec.models.config import Config

    config = Config.model_validate_json(config_json)
    emitter = _WorkerEventEmitter(args.control_socket_path)
    service = _RuntimeWorkerService(
        config=config,
        generation=args.generation,
        correlation_id=args.correlation_id,
        heartbeat_interval_s=args.heartbeat_interval_s,
        emitter=emitter,
    )
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        if args.test_ignore_term:
            loop.add_signal_handler(sig, lambda: None)
        else:
            loop.add_signal_handler(sig, stop_event.set)

    try:
        if args.test_mode:
            await service.run_test_mode(
                stop_event,
                fail_startup=args.test_fail_startup,
            )
        else:
            await service.run_runtime(stop_event)
    except Exception as exc:
        service.emit_error(exc)
        raise
    finally:
        emitter.close()


def main() -> None:
    args = _parse_args(sys.argv[1:])
    configure_logging(
        log_level="INFO",
        camera_name=f"runtime-worker-g{args.generation}",
    )
    try:
        asyncio.run(_run_worker(args))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.error("Runtime worker failed: %s", exc, exc_info=exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
