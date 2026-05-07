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


def test_camera_talk_status_drops_backend_reason_without_backend() -> None:
    """Public talk status should not expose backend reasons without a backend ID."""
    # Given: A custom source reports a backend reason without a selected backend
    payload = {
        "camera_name": "front",
        "enabled": True,
        "state": "error",
        "backend": None,
        "backend_reason": "selected rtsp://admin:secret@example.local/stream1",
    }

    # When: Parsing the public camera talk status model
    status = CameraTalkStatus.model_validate(payload)

    # Then: The standalone backend reason is dropped
    assert status.backend is None
    assert status.backend_reason is None


def test_camera_talk_status_drops_secret_bearing_backend_reason() -> None:
    """Public talk status should not expose secret-bearing backend reasons."""
    # Given: A custom source reports a safe backend ID with an unsafe reason
    payload = {
        "camera_name": "front",
        "enabled": True,
        "state": "error",
        "backend": "vendor_backend",
        "backend_reason": "selected rtsp://admin:secret@example.local/stream1",
    }

    # When: Parsing the public camera talk status model
    status = CameraTalkStatus.model_validate(payload)

    # Then: The backend ID is preserved but the unsafe reason is dropped
    assert status.backend == "vendor_backend"
    assert status.backend_reason is None


def test_camera_talk_status_drops_raw_sdp_backend_reason() -> None:
    """Public talk status should not expose raw SDP backend reasons."""
    # Given: A custom source reports a safe backend ID with an embedded SDP snippet
    payload = {
        "camera_name": "front",
        "enabled": True,
        "state": "error",
        "backend": "vendor_backend",
        "backend_reason": "raw SDP: a=crypto:1 inline:keymaterial",
    }

    # When: Parsing the public camera talk status model
    status = CameraTalkStatus.model_validate(payload)

    # Then: The backend ID is preserved but the raw SDP reason is dropped
    assert status.backend == "vendor_backend"
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


def test_worker_talk_status_payload_drops_backend_reason_without_backend() -> None:
    """Worker talk IPC should not forward backend reasons without backend IDs."""
    # Given: A worker payload includes a backend reason without a backend identifier
    payload = {
        "enabled": True,
        "state": "error",
        "backend": None,
        "backend_reason": "selected rtsp://admin:secret@example.local/stream1",
    }

    # When: Parsing the worker IPC payload
    status = WorkerTalkStatusPayload.model_validate(payload)

    # Then: The standalone backend reason is dropped before API mapping
    assert status.backend is None
    assert status.backend_reason is None


def test_worker_talk_status_payload_drops_secret_bearing_backend_reason() -> None:
    """Worker talk IPC should not forward secret-bearing backend reasons."""
    # Given: A worker payload includes a safe backend ID with an unsafe reason
    payload = {
        "enabled": True,
        "state": "error",
        "backend": "vendor_backend",
        "backend_reason": "selected Bearer secret-token",
    }

    # When: Parsing the worker IPC payload
    status = WorkerTalkStatusPayload.model_validate(payload)

    # Then: The backend ID is preserved but the unsafe reason is dropped
    assert status.backend == "vendor_backend"
    assert status.backend_reason is None


def test_worker_talk_status_payload_drops_raw_sdp_backend_reason() -> None:
    """Worker talk IPC should not forward raw SDP backend reasons."""
    # Given: A worker payload includes a safe backend ID with an embedded SDP snippet
    payload = {
        "enabled": True,
        "state": "error",
        "backend": "vendor_backend",
        "backend_reason": "SDP answer v=0 m=audio 0 RTP/AVP 0",
    }

    # When: Parsing the worker IPC payload
    status = WorkerTalkStatusPayload.model_validate(payload)

    # Then: The backend ID is preserved but the raw SDP reason is dropped
    assert status.backend == "vendor_backend"
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


def test_runtime_talk_stream_drops_backend_reason_without_backend() -> None:
    """Runtime talk stream ready metadata should require a safe backend ID."""
    # Given: A runtime stream is created with a backend reason but no backend ID
    # When: The stream model is initialized
    stream = RuntimeTalkStream(
        camera_name="front",
        session_id="tk_reason_only",
        input=TalkInputFormat(),
        reader=cast(asyncio.StreamReader, object()),
        writer=cast(asyncio.StreamWriter, object()),
        backend=None,
        backend_reason="selected rtsp://admin:secret@example.local/stream1",
    )

    # Then: The reason that would be sent in WebSocket ready is dropped
    assert stream.backend is None
    assert stream.backend_reason is None


def test_runtime_talk_stream_drops_secret_bearing_backend_reason() -> None:
    """Runtime talk stream ready metadata should not expose unsafe reasons."""
    # Given: A runtime stream is created with a safe backend ID and unsafe reason
    # When: The stream model is initialized
    stream = RuntimeTalkStream(
        camera_name="front",
        session_id="tk_secret_reason",
        input=TalkInputFormat(),
        reader=cast(asyncio.StreamReader, object()),
        writer=cast(asyncio.StreamWriter, object()),
        backend="vendor_backend",
        backend_reason="selected rtsp://admin:secret@example.local/stream1",
    )

    # Then: The backend ID remains but the unsafe reason is dropped
    assert stream.backend == "vendor_backend"
    assert stream.backend_reason is None


def test_runtime_talk_stream_drops_raw_sdp_backend_reason() -> None:
    """Runtime talk stream ready metadata should not expose raw SDP reasons."""
    # Given: A runtime stream is created with a safe backend ID and SDP reason
    # When: The stream model is initialized
    stream = RuntimeTalkStream(
        camera_name="front",
        session_id="tk_sdp_reason",
        input=TalkInputFormat(),
        reader=cast(asyncio.StreamReader, object()),
        writer=cast(asyncio.StreamWriter, object()),
        backend="vendor_backend",
        backend_reason="raw SDP: a=crypto:1 inline:keymaterial",
    )

    # Then: The backend ID remains but the raw SDP reason is dropped
    assert stream.backend == "vendor_backend"
    assert stream.backend_reason is None
