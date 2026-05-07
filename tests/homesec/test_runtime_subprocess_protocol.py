"""Tests for runtime subprocess wire payload compatibility."""

from __future__ import annotations

from homesec.models.talk import TalkCapabilityState, TalkState
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
