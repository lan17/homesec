from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from homesec.models.talk import TalkInputFormat, TalkRefusalReason, TalkState
from homesec.sources.rtsp.talk.manager import (
    TalkManager,
    TalkManagerError,
    TalkSessionOpenRequest,
    TalkSessionPrepareRequest,
)


@dataclass(slots=True)
class _FakeSession:
    session_id: str
    camera_name: str = "front"
    selected_codec: str = "PCMU/8000"
    frames: list[bytes] = field(default_factory=list)
    closed: bool = False

    async def write_pcm_frame(self, frame: bytes) -> None:
        self.frames.append(frame)

    async def close(self) -> None:
        self.closed = True


async def _open_fake_session(request: TalkSessionOpenRequest) -> _FakeSession:
    return _FakeSession(session_id=request.session_id)


def _manager(*, max_session_s: float = 60.0, idle_timeout_s: float = 60.0) -> TalkManager:
    return TalkManager(
        camera_name="front",
        enabled=True,
        supported_codecs=["PCMU/8000"],
        open_session_factory=_open_fake_session,
        max_session_s=max_session_s,
        idle_timeout_s=idle_timeout_s,
    )


@pytest.mark.asyncio
async def test_talk_manager_enforces_one_reserved_or_active_session() -> None:
    manager = _manager()

    first = await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))
    second = await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_2"))

    assert first.accepted is True
    assert first.session_id == "tk_1"
    assert second.accepted is False
    assert second.refusal_reason == TalkRefusalReason.SESSION_ALREADY_ACTIVE
    assert manager.status().state == TalkState.STARTING
    assert manager.status().active_session_id == "tk_1"

    await manager.stop_session("tk_1")
    third = await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_3"))
    assert third.accepted is True


@pytest.mark.asyncio
async def test_talk_manager_opens_and_writes_valid_pcm_frames() -> None:
    manager = _manager()
    frame = b"\x00" * TalkInputFormat().expected_bytes_per_frame

    await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))
    session = await manager.open_session(TalkSessionOpenRequest(session_id="tk_1"))
    await manager.write_pcm_frame("tk_1", frame)

    assert isinstance(session, _FakeSession)
    assert session.frames == [frame]
    assert manager.status().state == TalkState.ACTIVE
    assert manager.status().selected_codec == "PCMU/8000"

    stopped = await manager.stop_session("tk_1")
    assert stopped is True
    assert session.closed is True
    assert manager.status().state == TalkState.IDLE


@pytest.mark.asyncio
async def test_talk_manager_stop_returns_false_for_wrong_session() -> None:
    manager = _manager()

    await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))
    session = await manager.open_session(TalkSessionOpenRequest(session_id="tk_1"))

    stopped = await manager.stop_session("tk_missing")

    assert stopped is False
    assert session.closed is False
    assert manager.status().state == TalkState.ACTIVE

    await manager.stop_session("tk_1")


@pytest.mark.asyncio
async def test_talk_manager_stop_does_not_wait_for_slow_open() -> None:
    opened = asyncio.Event()
    release = asyncio.Event()
    opened_sessions: list[_FakeSession] = []

    async def _open_slow_session(request: TalkSessionOpenRequest) -> _FakeSession:
        opened.set()
        await release.wait()
        session = _FakeSession(session_id=request.session_id)
        opened_sessions.append(session)
        return session

    manager = TalkManager(
        camera_name="front",
        enabled=True,
        supported_codecs=["PCMU/8000"],
        open_session_factory=_open_slow_session,
        max_session_s=60.0,
        idle_timeout_s=60.0,
    )
    await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))
    open_task = asyncio.create_task(manager.open_session(TalkSessionOpenRequest(session_id="tk_1")))
    await opened.wait()

    stopped = await asyncio.wait_for(manager.stop_session("tk_1"), timeout=0.1)
    release.set()

    with pytest.raises(TalkManagerError) as error:
        await open_task

    assert stopped is True
    assert error.value.reason == TalkRefusalReason.RUNTIME_UNAVAILABLE
    assert manager.status().state == TalkState.IDLE
    assert opened_sessions[0].closed is True


@pytest.mark.asyncio
async def test_talk_manager_rejects_wrong_sized_pcm_frame() -> None:
    manager = _manager()

    await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))
    await manager.open_session(TalkSessionOpenRequest(session_id="tk_1"))

    with pytest.raises(TalkManagerError) as error:
        await manager.write_pcm_frame("tk_1", b"too-short")

    assert error.value.reason == TalkRefusalReason.INVALID_AUDIO_FRAME
    await manager.stop_session("tk_1")


@pytest.mark.asyncio
async def test_talk_manager_timeout_cleanup_releases_reserved_session() -> None:
    manager = _manager(max_session_s=0.01, idle_timeout_s=60.0)

    await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))
    await asyncio.sleep(0.05)

    assert manager.status().state == TalkState.IDLE
    retry = await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_2"))
    assert retry.accepted is True


@pytest.mark.asyncio
async def test_talk_manager_timeout_cleanup_closes_active_session() -> None:
    manager = _manager(max_session_s=0.01, idle_timeout_s=60.0)

    await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))
    session = await manager.open_session(TalkSessionOpenRequest(session_id="tk_1"))
    await asyncio.sleep(0.05)

    assert isinstance(session, _FakeSession)
    assert session.closed is True
    assert manager.status().state == TalkState.IDLE
