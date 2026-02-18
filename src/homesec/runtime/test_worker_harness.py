"""Deterministic subprocess worker harness used only by behavior tests."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import socket
import sys
from pathlib import Path

from homesec.models.config import Config
from homesec.runtime.subprocess_protocol import (
    WorkerCameraStatusPayload,
    WorkerEvent,
    WorkerEventType,
)

logger = logging.getLogger(__name__)

_NORMAL_SCENARIO = "normal"
_STARTUP_FAIL_SCENARIO = "startup-fail"
_IGNORE_TERM_SCENARIO = "ignore-term"


class _HarnessEventEmitter:
    """Sends worker events to the subprocess controller socket."""

    def __init__(self, socket_path: Path) -> None:
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._socket.connect(str(socket_path))

    def send(self, event: WorkerEvent) -> None:
        self._socket.send(event.model_dump_json().encode("utf-8"))

    def close(self) -> None:
        self._socket.close()


class _HarnessService:
    """Produces predictable worker lifecycle events for controller tests."""

    def __init__(
        self,
        *,
        config: Config,
        generation: int,
        correlation_id: str,
        heartbeat_interval_s: float,
        scenario: str,
        emitter: _HarnessEventEmitter,
    ) -> None:
        self._config = config
        self._generation = generation
        self._correlation_id = correlation_id
        self._heartbeat_interval_s = heartbeat_interval_s
        self._scenario = scenario
        self._emitter = emitter

    async def run(self, stop_event: asyncio.Event) -> None:
        if self._scenario == _STARTUP_FAIL_SCENARIO:
            self._emit_event(
                WorkerEventType.ERROR,
                message="runtime harness startup failure",
            )
            raise RuntimeError("runtime harness startup failure")

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
        message = str(exc).strip() or type(exc).__name__
        self._emit_event(WorkerEventType.ERROR, message=message[:512])

    async def _heartbeat_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self._heartbeat_interval_s,
                )
            except asyncio.TimeoutError:
                self._emit_event(WorkerEventType.HEARTBEAT)

    def _emit_event(self, event: WorkerEventType, *, message: str | None = None) -> None:
        payload = WorkerEvent(
            event=event,
            generation=self._generation,
            correlation_id=self._correlation_id,
            pid=os.getpid(),
            cameras=self._camera_payloads(),
            message=message,
        )
        self._emitter.send(payload)

    def _camera_payloads(self) -> dict[str, WorkerCameraStatusPayload]:
        payloads: dict[str, WorkerCameraStatusPayload] = {}
        for camera in self._config.cameras:
            payloads[camera.name] = WorkerCameraStatusPayload(
                healthy=camera.enabled,
                last_heartbeat=0.0 if camera.enabled else None,
            )
        return payloads


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HomeSec runtime test worker harness")
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--config-json-path", type=Path, required=True)
    parser.add_argument("--control-socket-path", type=Path, required=True)
    parser.add_argument("--correlation-id", type=str, required=True)
    parser.add_argument("--heartbeat-interval-s", type=float, default=0.05)
    parser.add_argument(
        "--harness-scenario",
        choices=(_NORMAL_SCENARIO, _STARTUP_FAIL_SCENARIO, _IGNORE_TERM_SCENARIO),
        default=_NORMAL_SCENARIO,
    )
    return parser.parse_args(argv)


async def _run_harness(args: argparse.Namespace) -> None:
    config = Config.model_validate_json(args.config_json_path.read_text(encoding="utf-8"))
    emitter = _HarnessEventEmitter(args.control_socket_path)
    service = _HarnessService(
        config=config,
        generation=args.generation,
        correlation_id=args.correlation_id,
        heartbeat_interval_s=args.heartbeat_interval_s,
        scenario=args.harness_scenario,
        emitter=emitter,
    )
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        if args.harness_scenario == _IGNORE_TERM_SCENARIO and sig == signal.SIGTERM:
            loop.add_signal_handler(sig, lambda: None)
        else:
            loop.add_signal_handler(sig, stop_event.set)

    try:
        await service.run(stop_event)
    except Exception as exc:
        if args.harness_scenario != _STARTUP_FAIL_SCENARIO:
            service.emit_error(exc)
        raise
    finally:
        emitter.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(sys.argv[1:])
    try:
        asyncio.run(_run_harness(args))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.error("Runtime test worker harness failed: %s", exc, exc_info=exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
