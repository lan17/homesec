from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from homesec.models.talk import (
    TalkCapabilityProbeResult,
    TalkCapabilityState,
    TalkInputFormat,
    TalkRefusalReason,
    TalkState,
)
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
async def test_talk_manager_refresh_status_probes_and_caches_capability() -> None:
    """Status refresh should expose discovered camera talk capability."""
    probe_calls = 0

    async def _probe() -> TalkCapabilityProbeResult:
        nonlocal probe_calls
        probe_calls += 1
        return TalkCapabilityProbeResult(
            capability=TalkCapabilityState.SUPPORTED,
            offered_codecs=["PCMA/8000"],
            selected_codec="PCMA/8000",
        )

    manager = TalkManager(
        camera_name="front",
        enabled=True,
        supported_codecs=["PCMU/8000", "PCMA/8000"],
        open_session_factory=_open_fake_session,
        max_session_s=60.0,
        idle_timeout_s=60.0,
        capability_probe_factory=_probe,
    )

    # Given: A manager with an unprobed camera capability
    initial = manager.status()

    # When: Refreshing status twice inside the capability cache window
    first = await manager.refresh_status()
    second = await manager.refresh_status()

    # Then: The probe result is cached and reflected in status
    assert initial.capability == TalkCapabilityState.UNKNOWN
    assert initial.state == TalkState.TEMPORARILY_UNAVAILABLE
    assert first.capability == TalkCapabilityState.SUPPORTED
    assert first.state == TalkState.IDLE
    assert first.offered_codecs == ["PCMA/8000"]
    assert first.selected_codec == "PCMA/8000"
    assert second.capability == TalkCapabilityState.SUPPORTED
    assert probe_calls == 1


@pytest.mark.asyncio
async def test_talk_manager_prepare_forces_probe_and_rejects_unsupported_codec() -> None:
    """Prepare should refuse when capability discovery finds no supported codec."""
    probe_calls = 0

    async def _probe() -> TalkCapabilityProbeResult:
        nonlocal probe_calls
        probe_calls += 1
        return TalkCapabilityProbeResult(
            capability=TalkCapabilityState.UNSUPPORTED_CODEC,
            offered_codecs=["OPUS/48000"],
            refusal_reason=TalkRefusalReason.UNSUPPORTED_CODEC,
            message="SDP sendonly audio has no preferred codec",
        )

    manager = TalkManager(
        camera_name="front",
        enabled=True,
        supported_codecs=["PCMU/8000"],
        open_session_factory=_open_fake_session,
        max_session_s=60.0,
        idle_timeout_s=60.0,
        capability_probe_factory=_probe,
    )

    # Given: A manager whose camera advertises only unsupported talk codecs
    # When: Preparing a talk session
    result = await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))

    # Then: No session is reserved and the refusal is codec-specific
    assert result.accepted is False
    assert result.refusal_reason == TalkRefusalReason.UNSUPPORTED_CODEC
    assert manager.status().state == TalkState.UNSUPPORTED
    assert manager.status().offered_codecs == ["OPUS/48000"]
    assert probe_calls == 1


@pytest.mark.asyncio
async def test_talk_manager_reports_policy_separately_from_effective_enabled() -> None:
    """Status should distinguish product disablement from per-camera policy."""
    manager = TalkManager(
        camera_name="front",
        enabled=False,
        policy_enabled=True,
        supported_codecs=[],
        open_session_factory=_open_fake_session,
        max_session_s=60.0,
        idle_timeout_s=60.0,
    )

    # Given: A manager disabled by effective runtime state while camera policy allows talk
    # When: Reading current talk status
    status = manager.status()

    # Then: Status preserves the per-camera policy distinction
    assert status.enabled is False
    assert status.policy_enabled is True
    assert status.state == TalkState.DISABLED


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
async def test_talk_manager_timeout_does_not_wait_for_slow_open() -> None:
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
        max_session_s=0.01,
        idle_timeout_s=60.0,
    )
    await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))
    open_task = asyncio.create_task(manager.open_session(TalkSessionOpenRequest(session_id="tk_1")))
    await opened.wait()

    await asyncio.sleep(0.05)
    assert manager.status().state == TalkState.IDLE

    release.set()
    with pytest.raises(TalkManagerError) as error:
        await open_task

    assert error.value.reason == TalkRefusalReason.RUNTIME_UNAVAILABLE
    assert opened_sessions[0].closed is True


@pytest.mark.asyncio
async def test_talk_manager_stop_waits_for_in_flight_write_before_close() -> None:
    write_started = asyncio.Event()
    release_write = asyncio.Event()

    @dataclass(slots=True)
    class _BlockingSession:
        session_id: str
        camera_name: str = "front"
        selected_codec: str = "PCMU/8000"
        frames: list[bytes] = field(default_factory=list)
        closed_count: int = 0

        async def write_pcm_frame(self, frame: bytes) -> None:
            write_started.set()
            await release_write.wait()
            self.frames.append(frame)

        async def close(self) -> None:
            self.closed_count += 1

    opened_session: _BlockingSession | None = None

    async def _open_blocking_session(request: TalkSessionOpenRequest) -> _BlockingSession:
        nonlocal opened_session
        opened_session = _BlockingSession(session_id=request.session_id)
        return opened_session

    manager = TalkManager(
        camera_name="front",
        enabled=True,
        supported_codecs=["PCMU/8000"],
        open_session_factory=_open_blocking_session,
        max_session_s=60.0,
        idle_timeout_s=60.0,
    )
    frame = b"\x33" * TalkInputFormat().expected_bytes_per_frame

    await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))
    await manager.open_session(TalkSessionOpenRequest(session_id="tk_1"))
    assert opened_session is not None

    write_task = asyncio.create_task(manager.write_pcm_frame("tk_1", frame))
    await write_started.wait()
    stop_task = asyncio.create_task(manager.stop_session("tk_1"))
    await asyncio.sleep(0)

    assert stop_task.done() is False
    assert opened_session.closed_count == 0

    release_write.set()
    await write_task
    assert await stop_task is True

    assert opened_session.frames == [frame]
    assert opened_session.closed_count == 1
    assert manager.status().state == TalkState.IDLE


@pytest.mark.asyncio
async def test_talk_manager_timeout_waits_for_in_flight_write_before_close() -> None:
    write_started = asyncio.Event()
    release_write = asyncio.Event()

    @dataclass(slots=True)
    class _BlockingSession:
        session_id: str
        camera_name: str = "front"
        selected_codec: str = "PCMU/8000"
        frames: list[bytes] = field(default_factory=list)
        closed_count: int = 0

        async def write_pcm_frame(self, frame: bytes) -> None:
            write_started.set()
            await release_write.wait()
            self.frames.append(frame)

        async def close(self) -> None:
            self.closed_count += 1

    opened_session: _BlockingSession | None = None

    async def _open_blocking_session(request: TalkSessionOpenRequest) -> _BlockingSession:
        nonlocal opened_session
        opened_session = _BlockingSession(session_id=request.session_id)
        return opened_session

    manager = TalkManager(
        camera_name="front",
        enabled=True,
        supported_codecs=["PCMU/8000"],
        open_session_factory=_open_blocking_session,
        max_session_s=0.01,
        idle_timeout_s=60.0,
    )
    frame = b"\x44" * TalkInputFormat().expected_bytes_per_frame

    await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))
    await manager.open_session(TalkSessionOpenRequest(session_id="tk_1"))
    assert opened_session is not None

    write_task = asyncio.create_task(manager.write_pcm_frame("tk_1", frame))
    await write_started.wait()
    await asyncio.sleep(0.05)

    assert opened_session.closed_count == 0

    release_write.set()
    await write_task
    await asyncio.sleep(0)

    assert opened_session.frames == [frame]
    assert opened_session.closed_count == 1
    assert manager.status().state == TalkState.IDLE


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
async def test_talk_manager_write_after_stop_fails_cleanly() -> None:
    manager = _manager()
    frame = b"\x55" * TalkInputFormat().expected_bytes_per_frame

    await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))
    await manager.open_session(TalkSessionOpenRequest(session_id="tk_1"))
    assert await manager.stop_session("tk_1") is True

    with pytest.raises(TalkManagerError) as error:
        await manager.write_pcm_frame("tk_1", frame)

    assert error.value.reason == TalkRefusalReason.RUNTIME_UNAVAILABLE


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
