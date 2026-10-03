"""Preview signaling crosses the existing supervised Unix IPC boundary."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest

from homesec.models.config import HLSPreviewConfig, PreviewConfig, WebRTCPreviewConfig
from homesec.models.preview import (
    PreviewAnswer,
    PreviewOffer,
    PreviewSessionAction,
    PreviewSessionRefusalReason,
)
from homesec.plugins import discover_all_plugins
from homesec.plugins.sources import load_source_plugin
from homesec.runtime.subprocess_controller import (
    SubprocessRuntimeController,
    SubprocessRuntimeHandle,
)
from homesec.runtime.subprocess_protocol import (
    WorkerCommand,
    WorkerCommandResult,
    WorkerCommandType,
)
from homesec.sources.rtsp.core import RTSPSource
from homesec.sources.rtsp.live_publisher import (
    LivePublisherRefusalReason,
    LivePublisherStartRefusal,
)
from tests.homesec.rtsp import test_webrtc_publisher
from tests.homesec.test_runtime_subprocess_controller import _make_config
from tests.homesec.test_runtime_worker import _make_config as _worker_config
from tests.homesec.test_runtime_worker import _make_service

helper = test_webrtc_publisher.helper


@pytest.fixture
def camera_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Camera command boundaries with controllable discovery and a steady motion stream."""
    directory = tmp_path / "camera-tools"
    directory.mkdir()
    probe_started = directory / "probe-started"
    release_probe = directory / "release-probe"
    ffprobe = directory / "ffprobe"
    ffprobe.write_text(
        f"#!{sys.executable}\n"
        "import json,time\nfrom pathlib import Path\n"
        f"Path({str(probe_started)!r}).touch()\n"
        f"while not Path({str(release_probe)!r}).exists(): time.sleep(0.01)\n"
        "print(json.dumps({'streams':["
        "{'codec_type':'video','codec_name':'h264','width':640,'height':480,'avg_frame_rate':'25/1'},"
        "{'codec_type':'audio','codec_name':'aac'}]}))\n"
    )
    ffprobe.chmod(0o700)
    ffmpeg = directory / "ffmpeg"
    ffmpeg.write_text(
        f"#!{sys.executable}\n"
        "import re,sys,time\nfrom pathlib import Path\n"
        "if '-y' in sys.argv: Path(sys.argv[-1]).write_bytes(b'preflight-clip')\n"
        "elif 'rawvideo' in sys.argv:\n"
        "    width,height=map(int,re.search(r'scale=(\\d+):(\\d+)',sys.argv[sys.argv.index('-vf')+1]).groups())\n"
        "    while True:\n"
        "        sys.stdout.buffer.write(bytes(width*height))\n"
        "        sys.stdout.buffer.flush()\n"
        "        time.sleep(0.1)\n"
    )
    ffmpeg.chmod(0o700)
    monkeypatch.setenv("PATH", str(directory) + os.pathsep + os.environ["PATH"])
    return probe_started, release_probe


async def _wait_for_source_ready(source: RTSPSource) -> None:
    deadline = time.monotonic() + 5
    while not source.is_healthy():
        assert time.monotonic() < deadline, "Camera preflight did not complete"
        await asyncio.sleep(0.01)


def _load_preview_source(tmp_path: Path, preview: PreviewConfig) -> RTSPSource:
    discover_all_plugins()
    source = load_source_plugin(
        "rtsp",
        {
            "rtsp_url": "rtsp://user:source-secret@camera/main",
            "output_dir": str(tmp_path / "recordings"),
            "stream": {"disable_hwaccel": True},
        },
        camera_name="front",
        __runtime_preview__=preview,
    )
    assert isinstance(source, RTSPSource)
    return source


@pytest.fixture
async def preview_source(
    helper: tuple[Path, Path], tmp_path: Path, camera_tools: tuple[Path, Path]
) -> AsyncIterator[RTSPSource]:
    source = _load_preview_source(
        tmp_path,
        PreviewConfig(
            enabled=True,
            backend="webrtc",
            config=WebRTCPreviewConfig(helper_path=str(helper[0]), advertised_ip="127.0.0.1"),
        ),
    )
    try:
        camera_tools[1].touch()
        await source.start()
        await _wait_for_source_ready(source)
        yield source
    finally:
        await source.shutdown()


@pytest.mark.asyncio
async def test_webrtc_waits_for_discovery_before_opening_audio_input(
    helper: tuple[Path, Path], tmp_path: Path, camera_tools: tuple[Path, Path]
) -> None:
    # Given: Background source startup is waiting for a camera probe that will discover AAC audio.
    source = _load_preview_source(
        tmp_path,
        PreviewConfig(
            enabled=True,
            backend="webrtc",
            config=WebRTCPreviewConfig(helper_path=str(helper[0]), advertised_ip="127.0.0.1"),
        ),
    )
    try:
        await source.start()
        deadline = time.monotonic() + 5
        while not camera_tools[0].exists():
            assert time.monotonic() < deadline, "Camera discovery did not start"
            await asyncio.sleep(0.01)
        assert not source.is_healthy()
        async with _worker_commands(source) as send:
            # When: Activation and negotiation arrive before the probe finishes.
            activation = source.ensure_preview_active()
            refused = await send(
                _command(
                    WorkerCommandType.PREVIEW_NEGOTIATE,
                    preview_offer=PreviewOffer(sdp="v=0"),
                    lease_expires_at=time.time() + 10,
                )
            )

            # Then: Both return a temporary refusal without opening any helper or preview input.
            assert isinstance(activation, LivePublisherStartRefusal)
            assert activation.reason == LivePublisherRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE
            assert refused.preview_session_refusal is not None
            assert (
                refused.preview_session_refusal.reason
                == PreviewSessionRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE
            )
            assert not helper[1].exists()

            # When: Discovery finishes and the caller retries through the same command socket.
            camera_tools[1].touch()
            await _wait_for_source_ready(source)
            accepted = await send(
                _command(
                    WorkerCommandType.PREVIEW_NEGOTIATE,
                    preview_offer=PreviewOffer(sdp="v=0"),
                    lease_expires_at=time.time() + 10,
                )
            )

            # Then: Its first input already includes discovered audio, without a manual restart.
            assert accepted.preview_answer is not None
            commands = [json.loads(raw) for raw in helper[1].read_text().splitlines()]
            starts = [command for command in commands if command["command"] == "start"]
            assert len(starts) == 1
            assert "libopus" in starts[0]["ffmpeg_args"]
    finally:
        camera_tools[1].touch()
        await source.shutdown()


@asynccontextmanager
async def _worker_commands(
    source: object | None, *, enabled: bool = True
) -> AsyncIterator[Callable[[WorkerCommand], Awaitable[WorkerCommandResult]]]:
    config = _worker_config(notifiers=[], source_backend="rtsp", preview_enabled=enabled)
    service = _make_service(config)
    service._runtime_bundle = cast(
        Any, SimpleNamespace(sources_by_camera={} if source is None else {"front": source})
    )
    with TemporaryDirectory(prefix="hs-preview-", dir="/tmp") as directory:
        socket_path = Path(directory) / "worker.sock"
        server = await asyncio.start_unix_server(
            service._handle_command_connection, path=str(socket_path)
        )

        async def send(command: WorkerCommand) -> WorkerCommandResult:
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            try:
                writer.write(command.model_dump_json().encode() + b"\n")
                await writer.drain()
                raw = await asyncio.wait_for(reader.readline(), timeout=3)
                result = WorkerCommandResult.model_validate_json(raw)
                assert result.command_id == command.command_id
                assert result.generation == command.generation
                assert result.correlation_id == command.correlation_id
                return result
            finally:
                writer.close()
                await writer.wait_closed()

        try:
            yield send
        finally:
            server.close()
            await server.wait_closed()


def _command(command: WorkerCommandType, **fields: object) -> WorkerCommand:
    return WorkerCommand.model_validate(
        {
            "command": command,
            "command_id": str(uuid4()),
            "generation": 1,
            "correlation_id": "test-correlation-id",
            "camera_name": "front",
            **fields,
        }
    )


@pytest.mark.asyncio
async def test_worker_socket_dispatches_source_owned_viewer_lifecycle(
    preview_source: RTSPSource, helper: tuple[Path, Path]
) -> None:
    # Given: The real command server owns a registry-created RTSP source and helper.
    deadline = time.time() + 20
    async with _worker_commands(preview_source) as send:
        # When: Two peers negotiate, the first leaves, and the survivor renews.
        one = await send(
            _command(
                WorkerCommandType.PREVIEW_NEGOTIATE,
                preview_offer=PreviewOffer(sdp="v=0\r\ns=first\r\n"),
                lease_expires_at=deadline,
            )
        )
        two = await send(
            _command(
                WorkerCommandType.PREVIEW_NEGOTIATE,
                preview_offer=PreviewOffer(sdp="v=0\r\ns=second\r\n"),
                lease_expires_at=deadline,
            )
        )
        assert one.preview_answer is not None and two.preview_answer is not None
        closed = await send(
            _command(
                WorkerCommandType.PREVIEW_CLOSE_SESSION, session_id=one.preview_answer.session_id
            )
        )
        renewed = await send(
            _command(
                WorkerCommandType.PREVIEW_RENEW_SESSION,
                session_id=two.preview_answer.session_id,
                lease_expires_at=deadline,
            )
        )

        # Then: Typed responses and helper calls preserve each peer and its absolute lease.
        assert closed.preview_session_action == PreviewSessionAction(accepted=True)
        assert renewed.preview_session_action == PreviewSessionAction(accepted=True)
        commands = [json.loads(raw) for raw in helper[1].read_text().splitlines()]
        assert [item["command"] for item in commands].count("start") == 1
        offers = [item for item in commands if item["command"] == "offer"]
        assert [item["sdp"] for item in offers] == ["v=0\r\ns=first\r\n", "v=0\r\ns=second\r\n"]
        assert [item["session_id"] for item in offers] == [
            one.preview_answer.session_id,
            two.preview_answer.session_id,
        ]
        assert all(item["lease_expires_at"] == deadline for item in offers)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,source_available", [(False, True), (True, False)])
async def test_worker_refuses_disabled_or_unavailable_preview_source(
    preview_source: RTSPSource,
    helper: tuple[Path, Path],
    enabled: bool,
    source_available: bool,
) -> None:
    # Given: Preview is disabled or no compatible source is available.
    async with _worker_commands(
        preview_source if source_available else None, enabled=enabled
    ) as send:
        # When: An offer arrives at the real command socket.
        result = await send(
            _command(
                WorkerCommandType.PREVIEW_NEGOTIATE,
                preview_offer=PreviewOffer(sdp="v=0"),
                lease_expires_at=time.time() + 10,
            )
        )
        # Then: The worker refuses without opening a media input.
        assert result.preview_session_refusal is not None
        assert (
            result.preview_session_refusal.reason
            == PreviewSessionRefusalReason.UNSUPPORTED_TRANSPORT
        )
        assert not helper[1].exists()


_SESSION_COMMANDS = [
    WorkerCommandType.PREVIEW_NEGOTIATE,
    WorkerCommandType.PREVIEW_RENEW_SESSION,
    WorkerCommandType.PREVIEW_CLOSE_SESSION,
]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", _SESSION_COMMANDS)
async def test_worker_refuses_incomplete_session_command(
    preview_source: RTSPSource, helper: tuple[Path, Path], command: WorkerCommandType
) -> None:
    # Given: A capable source with no active viewers.
    async with _worker_commands(preview_source) as send:
        # When: A typed command omits the fields required for its operation.
        result = await send(_command(command))
        # Then: The stable invalid-offer refusal crosses IPC without camera I/O.
        assert result.preview_session_refusal is not None
        assert result.preview_session_refusal.reason == PreviewSessionRefusalReason.INVALID_OFFER
        assert not helper[1].exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", _SESSION_COMMANDS)
async def test_hls_source_refuses_webrtc_session_commands(
    tmp_path: Path, command: WorkerCommandType
) -> None:
    # Given: A registered RTSP source configured for HLS playback.
    source = _load_preview_source(
        tmp_path, PreviewConfig(enabled=True, config=HLSPreviewConfig(storage_dir=tmp_path / "hls"))
    )
    try:
        async with _worker_commands(source) as send:
            # When: WebRTC signaling targets its source interface.
            result = await send(
                _command(
                    command,
                    preview_offer=PreviewOffer(sdp="v=0"),
                    session_id=str(uuid4()),
                    lease_expires_at=time.time() + 10,
                )
            )
            # Then: Every operation preserves the unsupported-transport refusal.
            assert result.preview_session_refusal is not None
            assert (
                result.preview_session_refusal.reason
                == PreviewSessionRefusalReason.UNSUPPORTED_TRANSPORT
            )
    finally:
        await source.shutdown()


@pytest.mark.asyncio
async def test_worker_source_exception_is_redacted_at_ipc_boundary(
    preview_source: RTSPSource, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Given: A source boundary raises an exception containing camera credentials and SDP.
    sensitive = "rtsp://user:private-password@camera SDP private-ice-secret"

    def failed_offer(offer: PreviewOffer, lease_expires_at: float) -> PreviewAnswer:
        raise RuntimeError(sensitive)

    monkeypatch.setattr(preview_source, "negotiate_preview", failed_offer)
    async with _worker_commands(preview_source) as send:
        # When: The worker executes the failing source operation.
        result = await send(
            _command(
                WorkerCommandType.PREVIEW_NEGOTIATE,
                preview_offer=PreviewOffer(sdp="v=0"),
                lease_expires_at=time.time() + 10,
            )
        )
        # Then: Only a stable refusal is returned or logged.
        assert result.preview_session_refusal is not None
        assert (
            result.preview_session_refusal.reason
            == PreviewSessionRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE
        )
        assert sensitive not in result.model_dump_json()
        assert "private-password" not in caplog.text
        assert "private-ice-secret" not in caplog.text


@pytest.mark.asyncio
async def test_signaling_commands_preserve_camera_session_and_authorization_deadline() -> None:
    # Given: A running generation and a deterministic worker command socket
    controller = SubprocessRuntimeController(command_timeout_s=1.0)
    runtime = cast(
        SubprocessRuntimeHandle,
        await controller.build_candidate(
            _make_config(watch_dir="/tmp/preview", source_backend="rtsp", preview_enabled=True), 1
        ),
    )
    runtime.process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(60)", start_new_session=True
    )
    runtime.last_heartbeat_at = datetime.now(UTC)
    commands: list[WorkerCommand] = []
    session_id = str(uuid4())

    async def respond(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        command = WorkerCommand.model_validate_json(await reader.readline())
        commands.append(command)
        result = WorkerCommandResult(
            command=command.command,
            command_id=command.command_id,
            correlation_id=command.correlation_id,
            generation=command.generation,
            camera_name=command.camera_name,
        )
        if command.command == WorkerCommandType.PREVIEW_NEGOTIATE:
            result.preview_answer = PreviewAnswer(session_id=session_id, sdp="v=0\r\ns=answer\r\n")
        else:
            result.preview_session_action = PreviewSessionAction(accepted=True)
        writer.write(result.model_dump_json().encode() + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(respond, path=str(runtime.command_socket_path))
    deadline = time.time() + 20
    try:
        # When: The API negotiates, renews, then closes one viewer
        answer = await controller.negotiate_preview(
            runtime, "front", offer=PreviewOffer(sdp="v=0"), lease_expires_at=deadline
        )
        renewed = await controller.renew_preview_session(
            runtime, "front", session_id=session_id, lease_expires_at=deadline
        )
        closed = await controller.close_preview_session(runtime, "front", session_id=session_id)

        # Then: SDP is typed, authorization is absolute, and peer close does not become camera stop
        assert isinstance(answer, PreviewAnswer) and answer.session_id == session_id
        assert isinstance(renewed, PreviewSessionAction) and renewed.accepted
        assert isinstance(closed, PreviewSessionAction) and closed.accepted
        assert [command.command for command in commands] == [
            WorkerCommandType.PREVIEW_NEGOTIATE,
            WorkerCommandType.PREVIEW_RENEW_SESSION,
            WorkerCommandType.PREVIEW_CLOSE_SESSION,
        ]
        assert all(command.camera_name == "front" for command in commands)
        assert commands[0].preview_offer is not None
        assert commands[0].preview_offer.sdp == "v=0"
        assert commands[0].lease_expires_at == commands[1].lease_expires_at == deadline
        assert commands[2].session_id == session_id
    finally:
        server.close()
        await server.wait_closed()
        await controller.shutdown_runtime(runtime)
