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
async def test_talk_manager_probe_failure_recovers_on_later_success() -> None:
    """Capability errors should be transient once a later camera probe succeeds."""
    probe_results: list[TalkCapabilityProbeResult | Exception] = [
        RuntimeError("camera backchannel timed out"),
        TalkCapabilityProbeResult(
            capability=TalkCapabilityState.SUPPORTED,
            offered_codecs=["PCMU/8000"],
            selected_codec="PCMU/8000",
        ),
    ]

    async def _probe() -> TalkCapabilityProbeResult:
        result = probe_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    manager = TalkManager(
        camera_name="front",
        enabled=True,
        supported_codecs=["PCMU/8000"],
        open_session_factory=_open_fake_session,
        max_session_s=60.0,
        idle_timeout_s=60.0,
        capability_probe_factory=_probe,
    )

    # Given: A talk-capable camera whose first capability probe fails transiently
    failed = await manager.refresh_status(force=True)

    # When: A later refresh succeeds
    recovered = await manager.refresh_status(force=True)

    # Then: The manager exposes the recovered capability instead of the stale error
    assert failed.capability == TalkCapabilityState.ERROR
    assert failed.state == TalkState.ERROR
    assert failed.last_error == "camera backchannel timed out"
    assert recovered.capability == TalkCapabilityState.SUPPORTED
    assert recovered.state == TalkState.IDLE
    assert recovered.last_error is None
    assert recovered.selected_codec == "PCMU/8000"


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
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
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
async def test_talk_manager_generates_session_id_when_prepare_omits_one() -> None:
    """Prepared sessions should get an opaque manager-generated id by default."""
    manager = _manager()

    # Given: A browser prepare request that does not provide a session id
    # When: Reserving a talk session
    prepared = await manager.prepare_session(TalkSessionPrepareRequest())

    # Then: The manager returns an opaque talk session id and tracks it as starting
    assert prepared.accepted is True
    assert prepared.session_id is not None
    assert prepared.session_id.startswith("tk_")
    assert manager.status().active_session_id == prepared.session_id
    await manager.stop_session(prepared.session_id)


@pytest.mark.asyncio
async def test_talk_manager_refresh_skips_probe_while_session_is_reserved() -> None:
    """Capability refresh should avoid camera probing while a session slot is held."""
    probe_calls = 0

    async def _probe() -> TalkCapabilityProbeResult:
        nonlocal probe_calls
        probe_calls += 1
        return TalkCapabilityProbeResult(
            capability=TalkCapabilityState.SUPPORTED,
            offered_codecs=["PCMU/8000"],
            selected_codec="PCMU/8000",
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

    # Given: A prepared session after an initial capability probe
    await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))

    # When: Status refresh is requested while the reservation is held
    status = await manager.refresh_status(force=True)

    # Then: The cached capability is returned without a second camera probe
    assert probe_calls == 1
    assert status.state == TalkState.STARTING
    assert status.capability == TalkCapabilityState.SUPPORTED
    await manager.stop_session("tk_1")


@pytest.mark.asyncio
async def test_talk_manager_rejects_open_without_matching_reservation() -> None:
    """Open should require the exact session id reserved by prepare."""
    open_calls: list[TalkSessionOpenRequest] = []

    async def _open_unexpected_session(request: TalkSessionOpenRequest) -> _FakeSession:
        open_calls.append(request)
        return _FakeSession(session_id=request.session_id)

    manager = TalkManager(
        camera_name="front",
        enabled=True,
        supported_codecs=["PCMU/8000"],
        open_session_factory=_open_unexpected_session,
        max_session_s=60.0,
        idle_timeout_s=60.0,
    )

    # Given: No prepared talk reservation for the requested session id
    # When: The runtime attempts to open a stream
    with pytest.raises(TalkManagerError) as error:
        await manager.open_session(TalkSessionOpenRequest(session_id="tk_missing"))

    # Then: The request is refused before the camera adapter is called
    assert error.value.reason == TalkRefusalReason.RUNTIME_UNAVAILABLE
    assert open_calls == []


@pytest.mark.asyncio
async def test_talk_manager_rejects_duplicate_open_for_active_session() -> None:
    """Open should be idempotently refused once the reserved session is active."""
    manager = _manager()

    # Given: An already active talk session
    await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))
    session = await manager.open_session(TalkSessionOpenRequest(session_id="tk_1"))

    # When: The runtime tries to open the same session again
    with pytest.raises(TalkManagerError) as error:
        await manager.open_session(TalkSessionOpenRequest(session_id="tk_1"))

    # Then: The duplicate open is refused without closing the active session
    assert error.value.reason == TalkRefusalReason.SESSION_ALREADY_ACTIVE
    assert isinstance(session, _FakeSession)
    assert session.closed is False
    await manager.stop_session("tk_1")


@pytest.mark.asyncio
async def test_talk_manager_open_failure_releases_reserved_session() -> None:
    """Failed camera session opens should not strand the one-session reservation."""

    async def _open_failing_session(request: TalkSessionOpenRequest) -> _FakeSession:
        _ = request
        raise TalkManagerError(
            "camera rejected SETUP",
            reason=TalkRefusalReason.CAMERA_BACKCHANNEL_FAILED,
        )

    manager = TalkManager(
        camera_name="front",
        enabled=True,
        supported_codecs=["PCMU/8000"],
        open_session_factory=_open_failing_session,
        max_session_s=60.0,
        idle_timeout_s=60.0,
    )

    # Given: A reserved talk session whose camera open fails
    first = await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))

    # When: Opening that session raises from the camera adapter
    with pytest.raises(TalkManagerError) as error:
        await manager.open_session(TalkSessionOpenRequest(session_id="tk_1"))

    # Then: The reservation is cleared and a later session can prepare cleanly
    assert first.accepted is True
    assert error.value.reason == TalkRefusalReason.CAMERA_BACKCHANNEL_FAILED
    assert manager.status().state == TalkState.IDLE
    retry = await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_2"))
    assert retry.accepted is True


@pytest.mark.asyncio
async def test_talk_manager_rejects_input_mismatch_before_opening_camera_session() -> None:
    """Browser input format must match the prepared session before camera I/O starts."""
    open_calls: list[TalkSessionOpenRequest] = []

    async def _open_unexpected_session(request: TalkSessionOpenRequest) -> _FakeSession:
        open_calls.append(request)
        return _FakeSession(session_id=request.session_id)

    manager = TalkManager(
        camera_name="front",
        enabled=True,
        supported_codecs=["PCMU/8000"],
        open_session_factory=_open_unexpected_session,
        max_session_s=60.0,
        idle_timeout_s=60.0,
    )

    # Given: A talk session prepared for the default browser PCM format
    await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))

    # When: The WebSocket attaches with a different input format
    with pytest.raises(TalkManagerError) as error:
        await manager.open_session(
            TalkSessionOpenRequest(
                session_id="tk_1",
                input=TalkInputFormat(sample_rate=8000),
            )
        )

    # Then: The manager refuses before opening a camera backchannel session
    assert error.value.reason == TalkRefusalReason.INVALID_AUDIO_FRAME
    assert open_calls == []
    assert manager.status().state == TalkState.STARTING
    await manager.stop_session("tk_1")


@pytest.mark.asyncio
async def test_talk_manager_opens_and_writes_valid_pcm_frames() -> None:
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
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
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
    manager = _manager()

    await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))
    session = await manager.open_session(TalkSessionOpenRequest(session_id="tk_1"))

    stopped = await manager.stop_session("tk_missing")

    assert stopped is False
    assert session.closed is False
    assert manager.status().state == TalkState.ACTIVE

    await manager.stop_session("tk_1")


@pytest.mark.asyncio
async def test_talk_manager_shutdown_closes_active_session() -> None:
    """Runtime shutdown should close the current camera talk session."""
    manager = _manager()

    # Given: An active talk session
    await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))
    session = await manager.open_session(TalkSessionOpenRequest(session_id="tk_1"))

    # When: The source/runtime shuts down
    await manager.shutdown()

    # Then: The active camera session is closed and manager state is idle
    assert isinstance(session, _FakeSession)
    assert session.closed is True
    assert manager.status().state == TalkState.IDLE


@pytest.mark.asyncio
async def test_talk_manager_stop_does_not_wait_for_slow_open() -> None:
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
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
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
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
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
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
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
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
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
    manager = _manager()

    await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))
    await manager.open_session(TalkSessionOpenRequest(session_id="tk_1"))

    with pytest.raises(TalkManagerError) as error:
        await manager.write_pcm_frame("tk_1", b"too-short")

    assert error.value.reason == TalkRefusalReason.INVALID_AUDIO_FRAME
    await manager.stop_session("tk_1")


@pytest.mark.asyncio
async def test_talk_manager_write_after_stop_fails_cleanly() -> None:
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
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
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
    manager = _manager(max_session_s=0.01, idle_timeout_s=60.0)

    await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))
    await asyncio.sleep(0.05)

    assert manager.status().state == TalkState.IDLE
    retry = await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_2"))
    assert retry.accepted is True


@pytest.mark.asyncio
async def test_talk_manager_timeout_cleanup_closes_active_session() -> None:
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
    manager = _manager(max_session_s=0.01, idle_timeout_s=60.0)

    await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))
    session = await manager.open_session(TalkSessionOpenRequest(session_id="tk_1"))
    await asyncio.sleep(0.05)

    assert isinstance(session, _FakeSession)
    assert session.closed is True
    assert manager.status().state == TalkState.IDLE


@pytest.mark.asyncio
async def test_talk_manager_idle_cleanup_closes_active_session_after_silence() -> None:
    """Idle sessions should close when the browser stops sending audio frames."""
    manager = _manager(max_session_s=60.0, idle_timeout_s=0.01)

    # Given: An active talk session with no audio frame activity
    await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_1"))
    session = await manager.open_session(TalkSessionOpenRequest(session_id="tk_1"))

    # When: The idle timeout elapses
    await asyncio.sleep(0.05)

    # Then: The session is closed and the talk slot is available again
    assert isinstance(session, _FakeSession)
    assert session.closed is True
    assert manager.status().state == TalkState.IDLE
    retry = await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_2"))
    assert retry.accepted is True
