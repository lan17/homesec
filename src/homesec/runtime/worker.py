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
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from homesec.config import resolve_env_var
from homesec.interfaces import Notifier
from homesec.logging_setup import configure_logging
from homesec.models.alert import Alert
from homesec.notifiers.multiplex import MultiplexNotifier, NotifierEntry
from homesec.plugins import discover_all_plugins
from homesec.plugins.alert_policies import load_alert_policy
from homesec.plugins.notifiers import load_notifier_plugin
from homesec.plugins.sources import load_source_plugin
from homesec.runtime.assembly import RuntimeAssembler
from homesec.runtime.bootstrap import (
    RuntimePersistenceStack,
    build_runtime_persistence_stack,
)
from homesec.runtime.errors import sanitize_runtime_error
from homesec.runtime.models import (
    CameraPreviewStartRefusal,
    CameraPreviewStatus,
    CameraPreviewStopResult,
    PreviewRefusalReason,
    PreviewState,
    RuntimeBundle,
)
from homesec.runtime.subprocess_protocol import (
    WorkerCameraStatusPayload,
    WorkerCommand,
    WorkerCommandErrorCode,
    WorkerCommandResult,
    WorkerCommandType,
    WorkerEvent,
    WorkerEventType,
    WorkerPreviewRefusalPayload,
    WorkerPreviewStatusPayload,
    WorkerPreviewStopPayload,
)
from homesec.sources.rtsp.live_publisher import (
    LivePublisherStartRefusal,
    LivePublisherStatus,
)

if TYPE_CHECKING:
    from homesec.interfaces import (
        AlertPolicy,
        ClipSource,
        EventStore,
        StateStore,
        StorageBackend,
    )
    from homesec.models.config import Config
    from homesec.repository import ClipRepository

logger = logging.getLogger(__name__)


@runtime_checkable
class _PreviewCapableSource(Protocol):
    """Structural protocol for camera sources that expose preview controls."""

    def preview_status(self) -> LivePublisherStatus: ...

    def ensure_preview_active(self) -> LivePublisherStatus | LivePublisherStartRefusal: ...

    def stop_preview(self) -> None: ...


class _NoopNotifier(Notifier):
    """No-op notifier used when no notifiers are configured."""

    async def send(self, alert: Alert) -> None:
        _ = alert
        logger.debug("NoopNotifier: alert suppressed (no notifiers configured)")

    async def ping(self) -> bool:
        return True

    async def shutdown(self, timeout: float | None = None) -> None:
        pass


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
        command_socket_path: Path,
        emitter: _WorkerEventEmitter,
    ) -> None:
        self._config = config
        self._generation = generation
        self._correlation_id = correlation_id
        self._heartbeat_interval_s = heartbeat_interval_s
        self._command_socket_path = command_socket_path
        self._emitter = emitter

        self._storage: StorageBackend | None = None
        self._state_store: StateStore | None = None
        self._event_store: EventStore | None = None
        self._repository: ClipRepository | None = None
        self._assembler: RuntimeAssembler | None = None
        self._runtime_bundle: RuntimeBundle | None = None
        self._command_server: asyncio.Server | None = None
        self._camera_configs = {camera.name: camera for camera in config.cameras}

    async def run_runtime(self, stop_event: asyncio.Event) -> None:
        started = False
        heartbeat_task: asyncio.Task[None] | None = None

        try:
            discover_all_plugins()
            self._command_socket_path.unlink(missing_ok=True)
            self._command_server = await asyncio.start_unix_server(
                self._handle_command_connection,
                path=str(self._command_socket_path),
            )

            persistence = await self._build_runtime_persistence_stack()
            self._storage = persistence.storage
            self._state_store = persistence.state_store
            self._event_store = persistence.event_store
            self._repository = persistence.repository
            self._assembler = RuntimeAssembler(
                storage=persistence.storage,
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

            command_server = self._command_server
            self._command_server = None
            if command_server is not None:
                command_server.close()
                await command_server.wait_closed()
            self._command_socket_path.unlink(missing_ok=True)

            await self._shutdown_runtime_stack()

            if started:
                self._emit_event(WorkerEventType.STOPPED)

    def emit_error(self, exc: Exception) -> None:
        self._emit_event(
            WorkerEventType.ERROR,
            message=sanitize_runtime_error(exc),
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

    async def _build_runtime_persistence_stack(self) -> RuntimePersistenceStack:
        """Build shared storage and persistence components for the worker runtime."""
        return await build_runtime_persistence_stack(
            self._config,
            resolve_env=resolve_env_var,
            missing_dsn_message=("Postgres DSN is required for runtime worker state_store backend"),
            event_store_unavailable_warning=(
                "Runtime worker event store unavailable (NoopEventStore returned); events dropped"
            ),
        )

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
            logger.info("No notifiers configured; notifications disabled")
            return _NoopNotifier(), entries
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

    async def _handle_command_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            raw_command = await reader.readline()
            if not raw_command:
                return

            try:
                command = WorkerCommand.model_validate_json(raw_command.decode("utf-8"))
            except Exception as exc:
                logger.warning("Dropping invalid runtime preview command: %s", exc, exc_info=True)
                return

            result = self._handle_command(command)
            writer.write(result.model_dump_json().encode("utf-8") + b"\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    def _handle_command(self, command: WorkerCommand) -> WorkerCommandResult:
        if command.generation != self._generation or command.correlation_id != self._correlation_id:
            return WorkerCommandResult(
                command=command.command,
                command_id=command.command_id,
                generation=self._generation,
                correlation_id=self._correlation_id,
                camera_name=command.camera_name,
                error_message="Runtime worker rejected preview command for another generation",
            )

        if command.camera_name not in self._camera_configs:
            return WorkerCommandResult(
                command=command.command,
                command_id=command.command_id,
                generation=self._generation,
                correlation_id=self._correlation_id,
                camera_name=command.camera_name,
                error_code=WorkerCommandErrorCode.CAMERA_NOT_FOUND,
                error_message=f"Camera '{command.camera_name}' not found",
            )

        match command.command:
            case WorkerCommandType.PREVIEW_STATUS:
                return WorkerCommandResult(
                    command=command.command,
                    command_id=command.command_id,
                    generation=self._generation,
                    correlation_id=self._correlation_id,
                    camera_name=command.camera_name,
                    status=self._preview_status_payload(command.camera_name),
                )
            case WorkerCommandType.PREVIEW_ENSURE_ACTIVE:
                outcome = self._ensure_preview_active(command.camera_name)
                if isinstance(outcome, CameraPreviewStartRefusal):
                    return WorkerCommandResult(
                        command=command.command,
                        command_id=command.command_id,
                        generation=self._generation,
                        correlation_id=self._correlation_id,
                        camera_name=command.camera_name,
                        refusal=WorkerPreviewRefusalPayload(
                            reason=outcome.reason,
                            message=outcome.message,
                        ),
                    )
                return WorkerCommandResult(
                    command=command.command,
                    command_id=command.command_id,
                    generation=self._generation,
                    correlation_id=self._correlation_id,
                    camera_name=command.camera_name,
                    status=self._preview_status_payload(command.camera_name, outcome),
                )
            case WorkerCommandType.PREVIEW_FORCE_STOP:
                stop_result = self._force_stop_preview(command.camera_name)
                return WorkerCommandResult(
                    command=command.command,
                    command_id=command.command_id,
                    generation=self._generation,
                    correlation_id=self._correlation_id,
                    camera_name=command.camera_name,
                    stop_result=WorkerPreviewStopPayload(
                        accepted=stop_result.accepted,
                        state=stop_result.state,
                    ),
                )
            case _:
                return WorkerCommandResult(
                    command=command.command,
                    command_id=command.command_id,
                    generation=self._generation,
                    correlation_id=self._correlation_id,
                    camera_name=command.camera_name,
                    error_message=f"Unsupported preview command: {command.command}",
                )

    def _emit_event(self, event_type: WorkerEventType, *, message: str | None = None) -> None:
        event = WorkerEvent(
            event=event_type,
            generation=self._generation,
            correlation_id=self._correlation_id,
            pid=os.getpid(),
            cameras=self._collect_camera_statuses(),
            previews=self._collect_preview_statuses(),
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

    def _collect_preview_statuses(self) -> dict[str, WorkerPreviewStatusPayload]:
        return {
            camera.name: self._preview_status_payload(camera.name)
            for camera in self._config.cameras
        }

    def _preview_status_payload(
        self,
        camera_name: str,
        status: CameraPreviewStatus | None = None,
    ) -> WorkerPreviewStatusPayload:
        preview_status = status or self._preview_status(camera_name)
        return WorkerPreviewStatusPayload(
            enabled=preview_status.enabled,
            state=preview_status.state,
            viewer_count=preview_status.viewer_count,
            degraded_reason=preview_status.degraded_reason,
            last_error=preview_status.last_error,
            idle_shutdown_at=preview_status.idle_shutdown_at,
        )

    def _preview_status(self, camera_name: str) -> CameraPreviewStatus:
        source = self._source_for_camera(camera_name)
        if not self._preview_enabled(camera_name, source):
            return CameraPreviewStatus(
                camera_name=camera_name,
                enabled=False,
                state=PreviewState.IDLE,
            )

        if source is None:
            return CameraPreviewStatus(
                camera_name=camera_name,
                enabled=True,
                state=PreviewState.IDLE,
            )

        preview_status = source.preview_status()
        return CameraPreviewStatus(
            camera_name=camera_name,
            enabled=True,
            state=PreviewState(preview_status.state.value),
            viewer_count=preview_status.viewer_count,
            degraded_reason=preview_status.degraded_reason,
            last_error=preview_status.last_error,
            idle_shutdown_at=preview_status.idle_shutdown_at,
        )

    def _ensure_preview_active(
        self,
        camera_name: str,
    ) -> CameraPreviewStatus | CameraPreviewStartRefusal:
        source = self._source_for_camera(camera_name)
        if not self._preview_enabled(camera_name, source):
            return CameraPreviewStartRefusal(
                camera_name=camera_name,
                reason=PreviewRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE,
                message="Preview is not enabled for this camera",
            )
        if source is None:
            return CameraPreviewStartRefusal(
                camera_name=camera_name,
                reason=PreviewRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE,
                message="Preview source is not available",
            )

        result = source.ensure_preview_active()
        if isinstance(result, LivePublisherStartRefusal):
            return CameraPreviewStartRefusal(
                camera_name=camera_name,
                reason=PreviewRefusalReason(result.reason.value),
                message=result.message,
            )
        return CameraPreviewStatus(
            camera_name=camera_name,
            enabled=True,
            state=PreviewState(result.state.value),
            viewer_count=result.viewer_count,
            degraded_reason=result.degraded_reason,
            last_error=result.last_error,
            idle_shutdown_at=result.idle_shutdown_at,
        )

    def _force_stop_preview(self, camera_name: str) -> CameraPreviewStopResult:
        source = self._source_for_camera(camera_name)
        if source is not None and self._preview_enabled(camera_name, source):
            source.stop_preview()
        return CameraPreviewStopResult(
            camera_name=camera_name,
            accepted=True,
            state=PreviewState.STOPPING,
        )

    def _source_for_camera(self, camera_name: str) -> _PreviewCapableSource | None:
        bundle = self._runtime_bundle
        if bundle is None:
            return None
        source = bundle.sources_by_camera.get(camera_name)
        if isinstance(source, _PreviewCapableSource):
            return source
        return None

    def _preview_enabled(
        self,
        camera_name: str,
        source: _PreviewCapableSource | None,
    ) -> bool:
        camera = self._camera_configs[camera_name]
        if not self._config.preview.enabled or not camera.enabled:
            return False
        if source is not None:
            return True
        return camera.source.backend == "rtsp"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HomeSec runtime worker")
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--config-json-path", type=Path, required=True)
    parser.add_argument("--control-socket-path", type=Path, required=True)
    parser.add_argument("--command-socket-path", type=Path, required=True)
    parser.add_argument("--correlation-id", type=str, required=True)
    parser.add_argument("--heartbeat-interval-s", type=float, default=2.0)
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
        command_socket_path=args.command_socket_path,
        emitter=emitter,
    )
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    try:
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
