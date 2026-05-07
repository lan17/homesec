"""Tests for runtime subprocess wire payload compatibility."""

from __future__ import annotations

import asyncio
from typing import cast

from homesec.models.talk import CameraTalkStatus, TalkCapabilityState, TalkInputFormat, TalkState
from homesec.runtime.models import RuntimeTalkStream
from homesec.runtime.subprocess_protocol import WorkerTalkStatusPayload


def test_worker_talk_status_payload_derives_capability_defaults() -> None:
    """Legacy worker payloads should derive new capability fields."""
    # Given: A legacy talk status payload without Phase 8 fields
    payload = {
        "enabled": True,
        "state": "idle",
    }

    # When: Parsing the worker IPC payload
    status = WorkerTalkStatusPayload.model_validate(payload)

    # Then: New fields preserve the legacy enabled/idle meaning
    assert status.policy_enabled is True
    assert status.capability == TalkCapabilityState.SUPPORTED
    assert status.state == TalkState.IDLE
    assert status.backend is None
    assert status.backend_reason is None


def test_camera_talk_status_sanitizes_unsafe_backend_diagnostics() -> None:
    """Public talk status should not expose unsafe backend identifiers."""
    # Given: A custom source reports a credential-bearing backend identifier
    payload = {
        "camera_name": "front",
        "enabled": True,
        "state": "error",
        "backend": "rtsp://admin:secret@example.local/stream1",
        "backend_reason": "selected rtsp://admin:secret@example.local/stream1",
    }

    # When: Parsing the public camera talk status model
    status = CameraTalkStatus.model_validate(payload)

    # Then: The unsafe backend diagnostic fields are dropped
    assert status.backend is None
    assert status.backend_reason is None


def test_worker_talk_status_payload_sanitizes_unsafe_backend_diagnostics() -> None:
    """Worker talk IPC should not forward unsafe backend identifiers."""
    # Given: A worker payload includes an unsafe backend identifier
    payload = {
        "enabled": True,
        "state": "error",
        "backend": "Bearer secret-token",
        "backend_reason": "selected Bearer secret-token",
    }

    # When: Parsing the worker IPC payload
    status = WorkerTalkStatusPayload.model_validate(payload)

    # Then: The unsafe backend diagnostic fields are dropped before API mapping
    assert status.backend is None
    assert status.backend_reason is None


def test_runtime_talk_stream_sanitizes_unsafe_backend_diagnostics() -> None:
    """Runtime talk stream ready metadata should not expose unsafe backend identifiers."""
    # Given: A runtime stream is created with unsafe backend diagnostics
    # When: The stream model is initialized
    stream = RuntimeTalkStream(
        camera_name="front",
        session_id="tk_unsafe",
        input=TalkInputFormat(),
        reader=cast(asyncio.StreamReader, object()),
        writer=cast(asyncio.StreamWriter, object()),
        backend="rtsp://admin:secret@example.local/stream1",
        backend_reason="selected rtsp://admin:secret@example.local/stream1",
    )

    # Then: The metadata that will be sent in WebSocket ready is dropped
    assert stream.backend is None
    assert stream.backend_reason is None
