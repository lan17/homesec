"""End-to-end tests for API push-to-talk streaming into a fake ONVIF camera."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import socket
import struct
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

import homesec.runtime.worker as worker_module
from homesec.api.server import create_app
from homesec.interfaces import AlertPolicy, ClipSource, Notifier, ObjectFilter, VLMAnalyzer
from homesec.models.config import (
    AlertPolicyConfig,
    CameraConfig,
    CameraSourceConfig,
    CameraTalkConfig,
    Config,
    FastAPIServerConfig,
    NotifierConfig,
    StateStoreConfig,
    StorageConfig,
    TalkConfig,
)
from homesec.models.filter import FilterConfig
from homesec.models.talk import CameraTalkStatus, TalkInputFormat, TalkRefusalReason
from homesec.models.vlm import VLMConfig
from homesec.notifiers.multiplex import NotifierEntry
from homesec.pipeline import ClipPipeline
from homesec.runtime.controller import RuntimeController
from homesec.runtime.errors import (
    TalkCameraNotFoundError,
    TalkRuntimeUnavailableError,
    TalkStreamOpenRefused,
)
from homesec.runtime.manager import RuntimeManager
from homesec.runtime.models import (
    CameraPreviewStartRefusal,
    CameraPreviewStatus,
    CameraPreviewStopResult,
    CameraTalkSessionPrepared,
    CameraTalkStartRefusal,
    CameraTalkStopResult,
    ManagedRuntime,
    RuntimeBundle,
    RuntimeTalkStream,
    config_signature,
)
from homesec.runtime.subprocess_protocol import (
    WorkerCommand,
    WorkerCommandErrorCode,
    WorkerCommandResult,
    WorkerCommandType,
)
from homesec.sources.rtsp.core import RTSPSource, RTSPSourceConfig
from homesec.sources.rtsp.talk.g711 import encode_pcma, encode_pcmu
from homesec.sources.rtsp.talk.resample import resample_pcm_s16le_mono
from homesec.sources.rtsp.talk.rtp import parse_rtp_header
from tests.homesec.ui_dist_stub import ensure_stub_ui_dist

_BACKCHANNEL_SDP = """v=0\r
o=- 0 0 IN IP4 127.0.0.1\r
s=HomeSec fake backchannel camera\r
t=0 0\r
m=video 0 RTP/AVP 96\r
a=recvonly\r
a=rtpmap:96 H264/90000\r
a=control:trackID=video\r
m=audio 0 RTP/AVP 0\r
a=sendonly\r
a=rtpmap:0 PCMU/8000\r
a=control:trackID=backchannel\r
"""

_UNSUPPORTED_CODEC_SDP = """v=0\r
o=- 0 0 IN IP4 127.0.0.1\r
s=HomeSec fake unsupported backchannel camera\r
t=0 0\r
m=audio 0 RTP/AVP 111\r
a=sendonly\r
a=rtpmap:111 OPUS/48000/2\r
a=control:trackID=backchannel\r
"""

_PCMA_BACKCHANNEL_SDP = """v=0\r
o=- 0 0 IN IP4 127.0.0.1\r
s=HomeSec fake PCMA backchannel camera\r
t=0 0\r
m=audio 0 RTP/AVP 8\r
a=sendonly\r
a=rtpmap:8 PCMA/8000\r
a=control:trackID=backchannel\r
"""


@dataclass(slots=True)
class _RTSPRequest:
    method: str
    uri: str
    headers: dict[str, str]


class _FakeRTSPBackchannelServer:
    def __init__(
        self,
        *,
        describe_status: int = 200,
        setup_status: int = 200,
        play_status: int = 200,
        sdp: str = _BACKCHANNEL_SDP,
    ) -> None:
        self.describe_status = describe_status
        self.setup_status = setup_status
        self.play_status = play_status
        self.sdp = sdp
        self.requests: list[_RTSPRequest] = []
        self.interleaved_before_play: list[tuple[int, bytes]] = []
        self.interleaved_after_play: list[tuple[int, bytes]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._listen_socket: socket.socket | None = None
        self._connections: list[socket.socket] = []
        self._thread: threading.Thread | None = None
        self._port: int | None = None

    @property
    def url(self) -> str:
        if self._port is None:
            raise RuntimeError("fake RTSP server is not started")
        return f"rtsp://127.0.0.1:{self._port}/Streaming/Channels/101"

    def start(self) -> None:
        listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_socket.bind(("127.0.0.1", 0))
        listen_socket.listen()
        listen_socket.settimeout(0.1)
        self._listen_socket = listen_socket
        self._port = int(listen_socket.getsockname()[1])
        self._thread = threading.Thread(target=self._serve, name="fake-rtsp-talk", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=1.0):
            raise TimeoutError("fake RTSP server did not start")

    def stop(self) -> None:
        self._stop.set()
        listen_socket = self._listen_socket
        if listen_socket is not None:
            listen_socket.close()
        with self._lock:
            connections = list(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)

    def wait_for_methods(self, expected: list[str], *, timeout_s: float = 2.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                methods = [request.method for request in self.requests]
            if methods == expected:
                return
            time.sleep(0.01)
        raise AssertionError(f"Expected RTSP methods {expected}, got {methods}")

    def wait_for_rtp_packets(self, count: int, *, timeout_s: float = 2.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                actual = len(self.interleaved_after_play)
            if actual >= count:
                return
            time.sleep(0.01)
        raise AssertionError(f"Expected at least {count} RTP packets, got {actual}")

    def _serve(self) -> None:
        self._ready.set()
        listen_socket = self._listen_socket
        if listen_socket is None:
            return
        while not self._stop.is_set():
            try:
                connection, _addr = listen_socket.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            connection.settimeout(None)
            with self._lock:
                self._connections.append(connection)
            try:
                self._handle_connection(connection)
            finally:
                with self._lock:
                    if connection in self._connections:
                        self._connections.remove(connection)
                connection.close()

    def _handle_connection(self, connection: socket.socket) -> None:
        play_seen = False
        try:
            while not self._stop.is_set():
                first = _recv_exact(connection, 1)
                if first == b"$":
                    channel = _recv_exact(connection, 1)[0]
                    length = int.from_bytes(_recv_exact(connection, 2), "big")
                    payload = _recv_exact(connection, length)
                    with self._lock:
                        target = (
                            self.interleaved_after_play
                            if play_seen
                            else self.interleaved_before_play
                        )
                        target.append((channel, payload))
                    continue

                request = self._read_request(connection, first)
                with self._lock:
                    self.requests.append(request)
                connection.sendall(self._response_for(request))
                if request.method == "PLAY" and self.describe_status == 200:
                    play_seen = True
                if request.method == "TEARDOWN":
                    return
        except (EOFError, OSError):
            return

    def _read_request(self, connection: socket.socket, first: bytes) -> _RTSPRequest:
        head = first + _recv_until(connection, b"\r\n\r\n")
        text = head.decode("iso-8859-1")
        lines = text.split("\r\n")
        method, uri, _version = lines[0].split(" ", maxsplit=2)
        headers: dict[str, str] = {}
        for line in lines[1:]:
            name, separator, value = line.partition(":")
            if separator:
                headers[name.lower()] = value.strip()
        content_length = int(headers.get("content-length", "0") or "0")
        if content_length:
            _recv_exact(connection, content_length)
        return _RTSPRequest(method=method, uri=uri, headers=headers)

    def _response_for(self, request: _RTSPRequest) -> bytes:
        cseq = request.headers.get("cseq", "1")
        if request.method == "DESCRIBE":
            if self.describe_status != 200:
                return _rtsp_response(
                    cseq=cseq,
                    status=self.describe_status,
                    reason="Option Not Supported",
                )
            return _rtsp_response(
                cseq=cseq,
                headers={"Content-Type": "application/sdp"},
                body=self.sdp.encode(),
            )
        if request.method == "SETUP":
            if self.setup_status != 200:
                return _rtsp_response(
                    cseq=cseq,
                    status=self.setup_status,
                    reason="Setup Failed",
                )
            return _rtsp_response(
                cseq=cseq,
                headers={
                    "Transport": "RTP/AVP/TCP;unicast;interleaved=0-1",
                    "Session": "homesec-talk;timeout=60",
                },
            )
        if request.method == "PLAY":
            if self.play_status != 200:
                return _rtsp_response(
                    cseq=cseq,
                    status=self.play_status,
                    reason="Play Failed",
                )
            return _rtsp_response(cseq=cseq, headers={"Session": "homesec-talk"})
        if request.method == "TEARDOWN":
            return _rtsp_response(cseq=cseq, headers={"Session": "homesec-talk"})
        return _rtsp_response(cseq=cseq, status=405, reason="Method Not Allowed")


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_until(connection: socket.socket, marker: bytes) -> bytes:
    data = bytearray()
    while not data.endswith(marker):
        chunk = connection.recv(1)
        if not chunk:
            raise EOFError
        data.extend(chunk)
    return bytes(data)


def _rtsp_response(
    *,
    cseq: str,
    status: int = 200,
    reason: str = "OK",
    headers: dict[str, str] | None = None,
    body: bytes = b"",
) -> bytes:
    merged = {"CSeq": cseq, "Content-Length": str(len(body))}
    if headers:
        merged.update(headers)
    lines = [f"RTSP/1.0 {status} {reason}"]
    lines.extend(f"{name}: {value}" for name, value in merged.items())
    return ("\r\n".join(lines) + "\r\n\r\n").encode("iso-8859-1") + body


@dataclass(slots=True)
class _HarnessRuntime:
    generation: int
    config: Config
    config_signature: str
    temp_dir: Path
    command_socket_path: Path
    service: Any
    sources: list[RTSPSource]
    loop: asyncio.AbstractEventLoop | None = None
    thread: threading.Thread | None = None
    started: threading.Event = field(default_factory=threading.Event)
    start_error: BaseException | None = None
    connections: list[asyncio.StreamWriter] = field(default_factory=list)


class _NoopEmitter:
    def send(self, event: object) -> None:
        _ = event

    def close(self) -> None:
        return None


class _WorkerHarnessController(RuntimeController):
    def __init__(self, tmp_path: Path) -> None:
        self._tmp_path = tmp_path
        self._runtimes: list[_HarnessRuntime] = []
        self.command_timeout_s = 2.0

    async def build_candidate(self, config: Config, generation: int) -> ManagedRuntime:
        _ = self._tmp_path
        temp_dir = Path(tempfile.mkdtemp(prefix=f"hspt-{generation}-", dir="/tmp"))
        command_socket_path = temp_dir / "commands.sock"
        sources = [_build_rtsp_source(config, output_dir=temp_dir / "recordings")]
        service = _make_worker_service(
            config=config,
            generation=generation,
            command_socket_path=command_socket_path,
            sources=sources,
        )
        runtime = _HarnessRuntime(
            generation=generation,
            config=config,
            config_signature=config_signature(config),
            temp_dir=temp_dir,
            command_socket_path=command_socket_path,
            service=service,
            sources=sources,
        )
        self._runtimes.append(runtime)
        return runtime

    async def start_runtime(self, runtime: ManagedRuntime) -> None:
        handle = _require_harness_runtime(runtime)
        loop = asyncio.new_event_loop()
        handle.loop = loop

        def _run_loop() -> None:
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._start_command_server(handle))
            except BaseException as exc:
                handle.start_error = exc
                handle.started.set()
                loop.close()
                return
            handle.started.set()
            loop.run_forever()
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

        handle.thread = threading.Thread(
            target=_run_loop,
            name=f"talk-worker-{handle.generation}",
            daemon=True,
        )
        handle.thread.start()
        started = await asyncio.to_thread(handle.started.wait, 2.0)
        if not started:
            raise TimeoutError("worker harness command server did not start")
        if handle.start_error is not None:
            raise RuntimeError("worker harness command server failed") from handle.start_error

    async def shutdown_runtime(self, runtime: ManagedRuntime) -> None:
        handle = _require_harness_runtime(runtime)
        await self._shutdown_handle(handle)

    async def shutdown_all(self) -> None:
        for runtime in list(self._runtimes):
            await self._shutdown_handle(runtime)

    async def get_preview_status(
        self,
        runtime: ManagedRuntime,
        camera_name: str,
    ) -> CameraPreviewStatus:
        _ = runtime, camera_name
        raise RuntimeError("preview is outside this talk E2E harness")

    async def ensure_preview_active(
        self,
        runtime: ManagedRuntime,
        camera_name: str,
    ) -> CameraPreviewStatus | CameraPreviewStartRefusal:
        _ = runtime, camera_name
        raise RuntimeError("preview is outside this talk E2E harness")

    async def force_stop_preview(
        self,
        runtime: ManagedRuntime,
        camera_name: str,
    ) -> CameraPreviewStopResult:
        _ = runtime, camera_name
        raise RuntimeError("preview is outside this talk E2E harness")

    async def note_preview_viewer_activity(
        self,
        runtime: ManagedRuntime,
        camera_name: str,
        *,
        viewer_id: str | None = None,
    ) -> None:
        _ = runtime, camera_name, viewer_id

    async def get_talk_status(
        self,
        runtime: ManagedRuntime,
        camera_name: str,
    ) -> CameraTalkStatus:
        result = await self._send_command(runtime, WorkerCommandType.TALK_STATUS, camera_name)
        if result.error_code == WorkerCommandErrorCode.CAMERA_NOT_FOUND:
            raise TalkCameraNotFoundError(camera_name)
        if result.talk_status is None:
            raise TalkRuntimeUnavailableError("Worker returned no talk status")
        return CameraTalkStatus(
            camera_name=camera_name,
            enabled=result.talk_status.enabled,
            policy_enabled=result.talk_status.policy_enabled,
            capability=result.talk_status.capability,
            state=result.talk_status.state,
            active_session_id=result.talk_status.active_session_id,
            supported_codecs=result.talk_status.supported_codecs,
            offered_codecs=result.talk_status.offered_codecs,
            selected_codec=result.talk_status.selected_codec,
            last_error=result.talk_status.last_error,
        )

    async def prepare_talk_session(
        self,
        runtime: ManagedRuntime,
        camera_name: str,
        *,
        session_id: str,
        input_format: TalkInputFormat,
    ) -> CameraTalkSessionPrepared | CameraTalkStartRefusal:
        result = await self._send_command(
            runtime,
            WorkerCommandType.TALK_PREPARE_SESSION,
            camera_name,
            session_id=session_id,
            talk_input=input_format,
        )
        if result.error_code == WorkerCommandErrorCode.CAMERA_NOT_FOUND:
            raise TalkCameraNotFoundError(camera_name)
        if result.talk_refusal is not None:
            return CameraTalkStartRefusal(
                camera_name=camera_name,
                reason=result.talk_refusal.reason,
                message=result.talk_refusal.message,
            )
        if result.talk_prepare is None:
            return CameraTalkStartRefusal(
                camera_name=camera_name,
                reason=TalkRefusalReason.RUNTIME_UNAVAILABLE,
                message="Worker returned no talk prepare result",
            )
        return CameraTalkSessionPrepared(
            camera_name=camera_name,
            session_id=result.talk_prepare.session_id,
            input=result.talk_prepare.input,
        )

    async def open_talk_stream(
        self,
        runtime: ManagedRuntime,
        camera_name: str,
        *,
        session_id: str,
        input_format: TalkInputFormat,
    ) -> RuntimeTalkStream:
        handle = _require_harness_runtime(runtime)
        command = WorkerCommand(
            command=WorkerCommandType.TALK_STREAM_OPEN,
            command_id=uuid.uuid4().hex,
            generation=handle.generation,
            correlation_id="talk-e2e",
            camera_name=camera_name,
            session_id=session_id,
            talk_input=input_format,
        )
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(handle.command_socket_path)),
            timeout=self.command_timeout_s,
        )
        close_writer = True
        try:
            writer.write(command.model_dump_json().encode("utf-8") + b"\n")
            await asyncio.wait_for(writer.drain(), timeout=self.command_timeout_s)
            raw_response = await asyncio.wait_for(
                reader.readline(),
                timeout=self.command_timeout_s,
            )
            if not raw_response:
                raise TalkRuntimeUnavailableError("Worker closed talk stream before response")
            result = WorkerCommandResult.model_validate_json(raw_response.decode("utf-8"))
            if result.error_code == WorkerCommandErrorCode.CAMERA_NOT_FOUND:
                raise TalkCameraNotFoundError(camera_name)
            if result.talk_refusal is not None:
                raise TalkStreamOpenRefused(
                    result.talk_refusal.message,
                    reason=result.talk_refusal.reason,
                )
            selected_codec = (
                result.talk_status.selected_codec if result.talk_status is not None else None
            )
            close_writer = False
            return RuntimeTalkStream(
                camera_name=camera_name,
                session_id=session_id,
                input=input_format,
                reader=reader,
                writer=writer,
                selected_codec=selected_codec,
            )
        finally:
            if close_writer:
                writer.close()
                await writer.wait_closed()

    async def stop_talk_session(
        self,
        runtime: ManagedRuntime,
        camera_name: str,
        *,
        session_id: str,
    ) -> CameraTalkStopResult:
        result = await self._send_command(
            runtime,
            WorkerCommandType.TALK_STOP_SESSION,
            camera_name,
            session_id=session_id,
        )
        if result.error_code == WorkerCommandErrorCode.CAMERA_NOT_FOUND:
            raise TalkCameraNotFoundError(camera_name)
        if result.talk_stop_result is None:
            raise TalkRuntimeUnavailableError("Worker returned no talk stop result")
        return CameraTalkStopResult(
            camera_name=camera_name,
            accepted=result.talk_stop_result.accepted,
            state=result.talk_stop_result.state,
        )

    async def _start_command_server(self, runtime: _HarnessRuntime) -> None:
        runtime.command_socket_path.unlink(missing_ok=True)

        async def _handle_connection(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            runtime.connections.append(writer)
            try:
                await runtime.service._handle_command_connection(reader, writer)
            finally:
                if writer in runtime.connections:
                    runtime.connections.remove(writer)

        runtime.service._command_server = await asyncio.start_unix_server(
            _handle_connection,
            path=str(runtime.command_socket_path),
        )

    async def _shutdown_handle(self, runtime: _HarnessRuntime) -> None:
        loop = runtime.loop
        thread = runtime.thread
        if loop is None or thread is None or loop.is_closed() or not thread.is_alive():
            shutil.rmtree(runtime.temp_dir, ignore_errors=True)
            return

        async def _shutdown_on_worker_loop() -> None:
            command_server = runtime.service._command_server
            runtime.service._command_server = None
            if command_server is not None:
                command_server.close()
                await command_server.wait_closed()
            for source in runtime.sources:
                await source.shutdown(timeout=1.0)
            for writer in list(runtime.connections):
                writer.close()
            for writer in list(runtime.connections):
                try:
                    await writer.wait_closed()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            runtime.command_socket_path.unlink(missing_ok=True)
            await asyncio.sleep(0)

        future = asyncio.run_coroutine_threadsafe(_shutdown_on_worker_loop(), loop)
        await asyncio.wrap_future(future)
        loop.call_soon_threadsafe(loop.stop)
        await asyncio.to_thread(thread.join, 2.0)
        shutil.rmtree(runtime.temp_dir, ignore_errors=True)

    async def _send_command(
        self,
        runtime: ManagedRuntime,
        command_type: WorkerCommandType,
        camera_name: str,
        *,
        session_id: str | None = None,
        talk_input: TalkInputFormat | None = None,
    ) -> WorkerCommandResult:
        handle = _require_harness_runtime(runtime)
        command = WorkerCommand(
            command=command_type,
            command_id=uuid.uuid4().hex,
            generation=handle.generation,
            correlation_id="talk-e2e",
            camera_name=camera_name,
            session_id=session_id,
            talk_input=talk_input,
        )
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(handle.command_socket_path)),
            timeout=self.command_timeout_s,
        )
        try:
            writer.write(command.model_dump_json().encode("utf-8") + b"\n")
            await asyncio.wait_for(writer.drain(), timeout=self.command_timeout_s)
            raw_response = await asyncio.wait_for(
                reader.readline(),
                timeout=self.command_timeout_s,
            )
            if not raw_response:
                raise TalkRuntimeUnavailableError("Worker closed command stream")
            return WorkerCommandResult.model_validate_json(raw_response.decode("utf-8"))
        finally:
            writer.close()
            await writer.wait_closed()


class _E2EApp:
    def __init__(self, config: Config, controller: _WorkerHarnessController) -> None:
        self._config = config
        self._manager = RuntimeManager(controller)
        self._shutdown = False
        self._bootstrap_mode = False

    @property
    def config(self) -> Config:
        return self._config

    @property
    def server_config(self) -> FastAPIServerConfig:
        return self._config.server

    @property
    def bootstrap_mode(self) -> bool:
        return self._bootstrap_mode

    async def start(self) -> None:
        await self._manager.start_initial_runtime(self._config)

    async def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        await self._manager.shutdown()

    async def get_camera_talk_status(self, camera_name: str) -> CameraTalkStatus:
        return await self._manager.get_talk_status(camera_name)

    async def prepare_camera_talk_session(
        self,
        camera_name: str,
        *,
        session_id: str,
        input_format: TalkInputFormat,
    ) -> CameraTalkSessionPrepared | CameraTalkStartRefusal:
        return await self._manager.prepare_talk_session(
            camera_name,
            session_id=session_id,
            input_format=input_format,
        )

    async def open_camera_talk_stream(
        self,
        camera_name: str,
        *,
        session_id: str,
        input_format: TalkInputFormat,
    ) -> RuntimeTalkStream:
        return await self._manager.open_talk_stream(
            camera_name,
            session_id=session_id,
            input_format=input_format,
        )

    async def stop_camera_talk_session(
        self,
        camera_name: str,
        *,
        session_id: str,
    ) -> CameraTalkStopResult:
        return await self._manager.stop_talk_session(camera_name, session_id=session_id)


def _require_harness_runtime(runtime: ManagedRuntime) -> _HarnessRuntime:
    if not isinstance(runtime, _HarnessRuntime):
        raise TypeError(f"Expected _HarnessRuntime, got {type(runtime).__name__}")
    return runtime


def _make_worker_service(
    *,
    config: Config,
    generation: int,
    command_socket_path: Path,
    sources: list[RTSPSource],
) -> Any:
    service = worker_module._RuntimeWorkerService(
        config=config,
        generation=generation,
        correlation_id="talk-e2e",
        heartbeat_interval_s=60.0,
        command_socket_path=command_socket_path,
        emitter=cast(Any, _NoopEmitter()),
    )
    service._runtime_bundle = RuntimeBundle(
        generation=generation,
        config=config,
        config_signature=config_signature(config),
        notifier=cast(Notifier, object()),
        notifier_entries=cast(list[NotifierEntry], []),
        filter_plugin=cast(ObjectFilter, object()),
        vlm_plugin=cast(VLMAnalyzer, object()),
        alert_policy=cast(AlertPolicy, object()),
        pipeline=cast(ClipPipeline, object()),
        sources=cast(list[ClipSource], sources),
        sources_by_camera={source.camera_name: cast(ClipSource, source) for source in sources},
    )
    return service


def _build_rtsp_source(config: Config, *, output_dir: Path) -> RTSPSource:
    camera = config.cameras[0]
    source_config = camera.source.config
    source_data = (
        source_config.model_dump(mode="python")
        if isinstance(source_config, BaseModel)
        else dict(source_config)
    )
    source_data["camera_name"] = camera.name
    source_data["output_dir"] = str(output_dir)
    source_data["__runtime_preview__"] = config.preview
    source_data["__runtime_talk__"] = config.talk
    source_data["__camera_talk__"] = camera.talk
    rtsp_source_config = RTSPSourceConfig.model_validate(source_data)
    return RTSPSource(config=rtsp_source_config, camera_name=camera.name)


def _make_config(
    *,
    tmp_path: Path,
    rtsp_url: str,
    input_format: TalkInputFormat | None = None,
) -> Config:
    resolved_input = input_format or TalkInputFormat(sample_rate=16000, frame_ms=20)
    return Config(
        cameras=[
            CameraConfig(
                name="front",
                source=CameraSourceConfig(
                    backend="rtsp",
                    config={
                        "rtsp_url": rtsp_url,
                        "output_dir": str(tmp_path / "recordings"),
                        "stream": {"disable_hwaccel": True},
                    },
                ),
                talk=CameraTalkConfig(
                    enabled=True,
                    config={"rtsp_url": rtsp_url, "connect_timeout_s": 1.0, "io_timeout_s": 1.0},
                ),
            )
        ],
        storage=StorageConfig(backend="local", config={"root": str(tmp_path / "storage")}),
        state_store=StateStoreConfig(dsn="postgresql://user:pass@localhost/homesec"),
        notifiers=[NotifierConfig(backend="noop", enabled=False, config={})],
        server=ensure_stub_ui_dist(FastAPIServerConfig()),
        talk=TalkConfig(
            enabled=True,
            max_session_s=5,
            idle_timeout_s=5.0,
            input=resolved_input,
        ),
        filter=FilterConfig(backend="yolo", config={}),
        vlm=VLMConfig(backend="openai", config={}),
        alert_policy=AlertPolicyConfig(backend="noop", config={}),
    )


def _start_message(input_format: TalkInputFormat) -> str:
    return json.dumps({"type": "start", **input_format.model_dump(mode="json")})


def _stop_message() -> str:
    return json.dumps({"type": "stop"})


def _pcm_frame(input_format: TalkInputFormat, value: int) -> bytes:
    sample_count = input_format.sample_rate * input_format.frame_ms // 1000
    return struct.pack(f"<{sample_count}h", *([value] * sample_count))


def _expected_pcmu_payload(input_format: TalkInputFormat, pcm_frame: bytes) -> bytes:
    resampled = resample_pcm_s16le_mono(
        pcm_frame,
        input_rate=input_format.sample_rate,
        output_rate=8000,
    )
    return encode_pcmu(resampled)


def _expected_pcma_payload(input_format: TalkInputFormat, pcm_frame: bytes) -> bytes:
    resampled = resample_pcm_s16le_mono(
        pcm_frame,
        input_rate=input_format.sample_rate,
        output_rate=8000,
    )
    return encode_pcma(resampled)


def _rtp_payload(packet: bytes) -> bytes:
    return packet[12:]


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_talk_status_discovers_fake_onvif_backchannel(tmp_path: Path) -> None:
    """Talk status should probe ONVIF backchannel capability before UI start."""
    server = _FakeRTSPBackchannelServer()
    app: _E2EApp | None = None

    try:
        server.start()
        config = _make_config(tmp_path=tmp_path, rtsp_url=server.url)
        app = _E2EApp(config, _WorkerHarnessController(tmp_path))
        _run(app.start())

        with TestClient(create_app(cast(Any, app))) as client:
            # Given: A running backend with global talk enabled for an RTSP camera
            # When: The UI asks for talk status
            response = client.get("/api/v1/talk/cameras/front")

            # Then: Status includes discovered camera capability and negotiated codec
            assert response.status_code == 200
            payload = response.json()
            assert payload["enabled"] is True
            assert payload["policy_enabled"] is True
            assert payload["capability"] == "supported"
            assert payload["state"] == "idle"
            assert payload["offered_codecs"] == ["PCMU/8000"]
            assert payload["selected_codec"] == "PCMU/8000"
        server.wait_for_methods(["DESCRIBE"])
    finally:
        if app is not None:
            _run(app.shutdown())
        server.stop()

    with server._lock:
        # Then: Capability discovery stops at DESCRIBE and does not open RTP.
        assert [request.method for request in server.requests] == ["DESCRIBE"]
        assert server.interleaved_before_play == []
        assert server.interleaved_after_play == []


def test_talk_websocket_streams_pcm_to_fake_onvif_backchannel(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Full API stream should open ONVIF backchannel and send RTP only after PLAY."""
    caplog.set_level(logging.DEBUG)
    server = _FakeRTSPBackchannelServer()
    app: _E2EApp | None = None

    try:
        server.start()
        secret_url = server.url.replace("rtsp://", "rtsp://alice:s3cr%40t@", 1)
        input_format = TalkInputFormat(sample_rate=16000, frame_ms=20)
        first_frame = _pcm_frame(input_format, 1000)
        second_frame = _pcm_frame(input_format, -1000)
        config = _make_config(tmp_path=tmp_path, rtsp_url=secret_url, input_format=input_format)
        app = _E2EApp(config, _WorkerHarnessController(tmp_path))
        _run(app.start())

        with TestClient(create_app(cast(Any, app))) as client:
            # Given: A prepared push-to-talk session exposed by the public API.
            prepare = client.post(
                "/api/v1/talk/cameras/front/sessions",
                json={
                    "session_id": "tk_e2e",
                    "input": input_format.model_dump(mode="json"),
                },
            )
            assert prepare.status_code == 201
            stream_url = prepare.json()["websocket_url"]

            # When: The browser WebSocket sends two fixed-size PCM frames and stops.
            with client.websocket_connect(stream_url) as websocket:
                websocket.send_text(_start_message(input_format))
                assert websocket.receive_json() == {
                    "type": "ready",
                    "camera_name": "front",
                    "session_id": "tk_e2e",
                    "input": input_format.model_dump(mode="json"),
                    "camera_codec": "PCMU/8000",
                }
                websocket.send_bytes(first_frame)
                websocket.send_bytes(second_frame)
                websocket.send_text(_stop_message())
                with pytest.raises(WebSocketDisconnect):
                    websocket.receive_text()

        server.wait_for_methods(["DESCRIBE", "DESCRIBE", "SETUP", "PLAY", "TEARDOWN"])
        server.wait_for_rtp_packets(2)
    finally:
        if app is not None:
            _run(app.shutdown())
        server.stop()

    with server._lock:
        requests = list(server.requests)
        before_play = list(server.interleaved_before_play)
        after_play = list(server.interleaved_after_play)

    # Then: The fake camera sees the ONVIF handshake and only post-PLAY RTP audio.
    assert [request.method for request in requests] == [
        "DESCRIBE",
        "DESCRIBE",
        "SETUP",
        "PLAY",
        "TEARDOWN",
    ]
    assert all(
        request.headers.get("require") == "www.onvif.org/ver20/backchannel"
        for request in requests[:4]
    )
    assert all("@" not in request.uri for request in requests)
    assert before_play == []
    assert len(after_play) == 2
    assert [channel for channel, _packet in after_play] == [0, 0]
    rtp_headers = [parse_rtp_header(packet) for _channel, packet in after_play]
    assert [header["payload_type"] for header in rtp_headers] == [0, 0]
    assert [_rtp_payload(packet) for _channel, packet in after_play] == [
        _expected_pcmu_payload(input_format, first_frame),
        _expected_pcmu_payload(input_format, second_frame),
    ]
    assert (
        rtp_headers[1]["sequence_number"] == (int(rtp_headers[0]["sequence_number"]) + 1) & 0xFFFF
    )
    assert (
        rtp_headers[1]["timestamp"]
        == (int(rtp_headers[0]["timestamp"]) + len(_rtp_payload(after_play[0][1]))) & 0xFFFFFFFF
    )
    assert rtp_headers[1]["ssrc"] == rtp_headers[0]["ssrc"]
    assert "s3cr" not in caplog.text
    assert _pcm_frame(input_format, 1000).hex() not in caplog.text


def test_talk_websocket_negotiates_pcma_backchannel(tmp_path: Path) -> None:
    """Full API stream should use PCMA when that is the camera's advertised G.711 codec."""
    server = _FakeRTSPBackchannelServer(sdp=_PCMA_BACKCHANNEL_SDP)
    app: _E2EApp | None = None

    try:
        server.start()
        input_format = TalkInputFormat(sample_rate=16000, frame_ms=20)
        pcm_frame = _pcm_frame(input_format, 1000)
        config = _make_config(tmp_path=tmp_path, rtsp_url=server.url, input_format=input_format)
        app = _E2EApp(config, _WorkerHarnessController(tmp_path))
        _run(app.start())

        with TestClient(create_app(cast(Any, app))) as client:
            # Given: A prepared push-to-talk session targeting a PCMA-only camera.
            prepare = client.post(
                "/api/v1/talk/cameras/front/sessions",
                json={
                    "session_id": "tk_pcma",
                    "input": input_format.model_dump(mode="json"),
                },
            )
            assert prepare.status_code == 201

            # When: The browser stream opens and sends one PCM frame.
            with client.websocket_connect(prepare.json()["websocket_url"]) as websocket:
                websocket.send_text(_start_message(input_format))
                assert websocket.receive_json()["camera_codec"] == "PCMA/8000"
                websocket.send_bytes(pcm_frame)
                websocket.send_text(_stop_message())
                with pytest.raises(WebSocketDisconnect):
                    websocket.receive_text()

        server.wait_for_methods(["DESCRIBE", "DESCRIBE", "SETUP", "PLAY", "TEARDOWN"])
        server.wait_for_rtp_packets(1)
    finally:
        if app is not None:
            _run(app.shutdown())
        server.stop()

    with server._lock:
        after_play = list(server.interleaved_after_play)

    # Then: The camera receives static PCMA RTP payload type 8 with A-law bytes.
    assert len(after_play) == 1
    _channel, packet = after_play[0]
    assert parse_rtp_header(packet)["payload_type"] == 8
    assert _rtp_payload(packet) == _expected_pcma_payload(input_format, pcm_frame)


def test_talk_websocket_disconnect_tears_down_fake_camera_session(tmp_path: Path) -> None:
    """Client disconnect should release the active ONVIF backchannel session."""
    server = _FakeRTSPBackchannelServer()
    app: _E2EApp | None = None

    try:
        server.start()
        input_format = TalkInputFormat(sample_rate=16000, frame_ms=20)
        config = _make_config(tmp_path=tmp_path, rtsp_url=server.url, input_format=input_format)
        app = _E2EApp(config, _WorkerHarnessController(tmp_path))
        _run(app.start())

        with TestClient(create_app(cast(Any, app))) as client:
            # Given: A prepared API session attached to a fake camera backchannel.
            prepare = client.post(
                "/api/v1/talk/cameras/front/sessions",
                json={
                    "session_id": "tk_disconnect",
                    "input": input_format.model_dump(mode="json"),
                },
            )
            assert prepare.status_code == 201
            stream_url = prepare.json()["websocket_url"]

            # When: The WebSocket drops without an explicit stop control message.
            with client.websocket_connect(stream_url) as websocket:
                websocket.send_text(_start_message(input_format))
                websocket.receive_json()
                websocket.send_bytes(_pcm_frame(input_format, 500))

        server.wait_for_methods(["DESCRIBE", "DESCRIBE", "SETUP", "PLAY", "TEARDOWN"])
    finally:
        if app is not None:
            _run(app.shutdown())
        server.stop()

    with server._lock:
        # Then: The worker/source cleanup sent TEARDOWN and did not emit pre-PLAY RTP.
        assert [request.method for request in server.requests] == [
            "DESCRIBE",
            "DESCRIBE",
            "SETUP",
            "PLAY",
            "TEARDOWN",
        ]
        assert server.interleaved_before_play == []
        assert len(server.interleaved_after_play) == 1


def test_talk_websocket_maps_fake_camera_unsupported_backchannel_to_policy_close(
    tmp_path: Path,
) -> None:
    """RTSP protocol failures should keep typed refusal reasons across the API stream."""
    server = _FakeRTSPBackchannelServer(describe_status=551)
    app: _E2EApp | None = None

    try:
        server.start()
        input_format = TalkInputFormat(sample_rate=16000, frame_ms=20)
        config = _make_config(tmp_path=tmp_path, rtsp_url=server.url, input_format=input_format)
        app = _E2EApp(config, _WorkerHarnessController(tmp_path))
        _run(app.start())

        with TestClient(create_app(cast(Any, app))) as client:
            # Given: A camera that rejects the ONVIF backchannel probe.
            prepare = client.post(
                "/api/v1/talk/cameras/front/sessions",
                json={
                    "session_id": "tk_unsupported",
                    "input": input_format.model_dump(mode="json"),
                },
            )
            # Then: Prepare refuses before creating a stream token.
            assert prepare.status_code == 409
            assert prepare.json()["error_code"] == "TALK_UNSUPPORTED_CAMERA"
            assert prepare.json()["reason"] == "unsupported_camera"
        server.wait_for_methods(["DESCRIBE"])
    finally:
        if app is not None:
            _run(app.shutdown())
        server.stop()

    with server._lock:
        assert [request.method for request in server.requests] == ["DESCRIBE"]
        assert server.interleaved_before_play == []
        assert server.interleaved_after_play == []


def test_talk_websocket_maps_fake_camera_unsupported_codec_to_policy_close(
    tmp_path: Path,
) -> None:
    """Unsupported camera speaker codecs should remain typed policy refusals."""
    server = _FakeRTSPBackchannelServer(sdp=_UNSUPPORTED_CODEC_SDP)
    app: _E2EApp | None = None

    try:
        server.start()
        input_format = TalkInputFormat(sample_rate=16000, frame_ms=20)
        config = _make_config(tmp_path=tmp_path, rtsp_url=server.url, input_format=input_format)
        app = _E2EApp(config, _WorkerHarnessController(tmp_path))
        _run(app.start())

        with TestClient(create_app(cast(Any, app))) as client:
            # Given: A camera that advertises only unsupported talk codecs.
            prepare = client.post(
                "/api/v1/talk/cameras/front/sessions",
                json={
                    "session_id": "tk_codec",
                    "input": input_format.model_dump(mode="json"),
                },
            )
            # Then: Prepare refuses with a codec-specific error.
            assert prepare.status_code == 400
            assert prepare.json()["error_code"] == "TALK_UNSUPPORTED_CODEC"
            assert prepare.json()["reason"] == "unsupported_codec"
        server.wait_for_methods(["DESCRIBE"])
    finally:
        if app is not None:
            _run(app.shutdown())
        server.stop()

    with server._lock:
        assert [request.method for request in server.requests] == ["DESCRIBE"]
        assert server.interleaved_before_play == []
        assert server.interleaved_after_play == []


def test_talk_websocket_maps_fake_camera_rejected_session_to_unavailable_close(
    tmp_path: Path,
) -> None:
    """Generic RTSP backchannel failures should remain retryable runtime refusals."""
    server = _FakeRTSPBackchannelServer(describe_status=500)
    app: _E2EApp | None = None

    try:
        server.start()
        input_format = TalkInputFormat(sample_rate=16000, frame_ms=20)
        config = _make_config(tmp_path=tmp_path, rtsp_url=server.url, input_format=input_format)
        app = _E2EApp(config, _WorkerHarnessController(tmp_path))
        _run(app.start())

        with TestClient(create_app(cast(Any, app))) as client:
            # Given: A camera whose backchannel probe fails transiently.
            prepare = client.post(
                "/api/v1/talk/cameras/front/sessions",
                json={
                    "session_id": "tk_rejected",
                    "input": input_format.model_dump(mode="json"),
                },
            )
            # Then: Prepare reports a retryable camera backchannel failure.
            assert prepare.status_code == 503
            assert prepare.json()["error_code"] == "TALK_CAMERA_BACKCHANNEL_FAILED"
            assert prepare.json()["reason"] == "camera_backchannel_failed"
        server.wait_for_methods(["DESCRIBE"])
    finally:
        if app is not None:
            _run(app.shutdown())
        server.stop()

    with server._lock:
        assert [request.method for request in server.requests] == ["DESCRIBE"]
        assert server.interleaved_before_play == []
        assert server.interleaved_after_play == []
