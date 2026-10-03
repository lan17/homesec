"""Preview signaling crosses the existing supervised Unix IPC boundary."""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import pytest

from homesec.models.preview import PreviewAnswer, PreviewOffer, PreviewSessionAction
from homesec.runtime.subprocess_controller import (
    SubprocessRuntimeController,
    SubprocessRuntimeHandle,
)
from homesec.runtime.subprocess_protocol import (
    WorkerCommand,
    WorkerCommandResult,
    WorkerCommandType,
)
from tests.homesec.test_runtime_subprocess_controller import _make_config


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
        assert commands[0].preview_offer.sdp == "v=0"
        assert commands[0].lease_expires_at == commands[1].lease_expires_at == deadline
        assert commands[2].session_id == session_id
    finally:
        server.close()
        await server.wait_closed()
        await controller.shutdown_runtime(runtime)
