"""Source-side talk session lifecycle manager."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from homesec.models.talk import (
    CameraTalkStatus,
    TalkInputFormat,
    TalkRefusalReason,
    TalkSessionOpenRequest,
    TalkSessionPrepareRequest,
    TalkSessionPrepareResult,
    TalkState,
)

logger = logging.getLogger(__name__)


class TalkSession(Protocol):
    """Camera talk session opened by a source adapter."""

    session_id: str
    camera_name: str

    @property
    def selected_codec(self) -> str:
        """Codec selected by the camera adapter for this session."""
        ...

    async def write_pcm_frame(self, frame: bytes) -> None:
        """Send one PCM frame to the camera speaker."""
        ...

    async def close(self) -> None:
        """Close the camera talk session."""
        ...


@dataclass(slots=True)
class TalkStats:
    """Transient talk session counters."""

    bytes_sent: int = 0
    frames_sent: int = 0
    dropped_frames: int = 0
    underruns: int = 0
    start_time: float | None = None
    last_frame_time: float | None = None
    selected_codec: str | None = None
    close_reason: str | None = None


@dataclass(slots=True)
class _SessionRecord:
    session_id: str
    input: TalkInputFormat
    state: TalkState
    created_at: float
    last_activity_at: float
    session: TalkSession | None = None
    opening: bool = False
    selected_codec: str | None = None
    stats: TalkStats = field(default_factory=TalkStats)
    timeout_task: asyncio.Task[None] | None = None
    idle_task: asyncio.Task[None] | None = None


class TalkManager:
    """Enforce one active/reserved talk session for a camera and clean it up."""

    def __init__(
        self,
        *,
        camera_name: str,
        enabled: bool,
        supported_codecs: list[str],
        open_session_factory: Callable[[TalkSessionOpenRequest], Awaitable[TalkSession]],
        max_session_s: float,
        idle_timeout_s: float,
    ) -> None:
        self._camera_name = camera_name
        self._enabled = enabled
        self._supported_codecs = list(supported_codecs)
        self._open_session_factory = open_session_factory
        self._max_session_s = max_session_s
        self._idle_timeout_s = idle_timeout_s
        self._record: _SessionRecord | None = None
        self._lock = asyncio.Lock()
        self._last_error: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def status(self) -> CameraTalkStatus:
        """Return current source-level talk status."""
        record = self._record
        if not self._enabled:
            state = TalkState.DISABLED
        elif record is None:
            state = TalkState.IDLE
        else:
            state = record.state
        return CameraTalkStatus(
            camera_name=self._camera_name,
            enabled=self._enabled,
            state=state,
            active_session_id=record.session_id if record is not None else None,
            supported_codecs=self._supported_codecs,
            selected_codec=record.selected_codec if record is not None else None,
            last_error=self._last_error,
        )

    async def prepare_session(
        self,
        request: TalkSessionPrepareRequest,
    ) -> TalkSessionPrepareResult:
        """Reserve a talk slot before the browser attaches the WebSocket stream."""
        self._bind_loop()
        async with self._lock:
            if not self._enabled:
                return TalkSessionPrepareResult(
                    accepted=False,
                    refusal_reason=TalkRefusalReason.TALK_DISABLED,
                    message="Talk is disabled for this camera",
                    input=request.input,
                )
            if self._record is not None:
                return TalkSessionPrepareResult(
                    accepted=False,
                    refusal_reason=TalkRefusalReason.SESSION_ALREADY_ACTIVE,
                    message="A talk session is already active for this camera",
                    input=request.input,
                )

            now = time.monotonic()
            session_id = request.session_id or _new_session_id()
            record = _SessionRecord(
                session_id=session_id,
                input=request.input,
                state=TalkState.STARTING,
                created_at=now,
                last_activity_at=now,
            )
            record.timeout_task = asyncio.create_task(self._max_duration_watch(record.session_id))
            self._record = record
            return TalkSessionPrepareResult(
                accepted=True,
                session_id=session_id,
                input=request.input,
            )

    async def open_session(self, request: TalkSessionOpenRequest) -> TalkSession:
        """Open the camera session for a reserved talk session id."""
        self._bind_loop()
        async with self._lock:
            record = self._record
            if record is None or record.session_id != request.session_id:
                raise TalkManagerError(
                    "Talk session is not reserved",
                    reason=TalkRefusalReason.RUNTIME_UNAVAILABLE,
                )
            if record.session is not None or record.opening:
                raise TalkManagerError(
                    "Talk session is already active",
                    reason=TalkRefusalReason.SESSION_ALREADY_ACTIVE,
                )
            if request.input != record.input:
                raise TalkManagerError(
                    "Talk input format does not match the reserved session",
                    reason=TalkRefusalReason.INVALID_AUDIO_FRAME,
                )
            record.opening = True

        try:
            session = await self._open_session_factory(request)
        except asyncio.CancelledError:
            await self._clear_opening_record(request.session_id, close_reason="open_cancelled")
            raise
        except Exception as exc:
            self._last_error = str(exc) or type(exc).__name__
            await self._clear_opening_record(request.session_id, close_reason="open_failed")
            raise

        async with self._lock:
            record = self._record
            if record is None or record.session_id != request.session_id:
                close_unclaimed_session = True
            else:
                close_unclaimed_session = False
                record.opening = False
                record.session = session
                record.state = TalkState.ACTIVE
                record.selected_codec = session.selected_codec
                record.stats.start_time = time.monotonic()
                record.stats.selected_codec = session.selected_codec
                if record.timeout_task is None:
                    record.timeout_task = asyncio.create_task(
                        self._max_duration_watch(record.session_id)
                    )
                record.idle_task = asyncio.create_task(self._idle_watch(record.session_id))
                return session

        if close_unclaimed_session:
            await _close_session_safely(session)
            raise TalkManagerError(
                "Talk session is no longer reserved",
                reason=TalkRefusalReason.RUNTIME_UNAVAILABLE,
            )

        raise AssertionError("unreachable talk session open state")

    async def write_pcm_frame(self, session_id: str, frame: bytes) -> None:
        """Validate and forward one PCM input frame to the active camera session."""
        self._bind_loop()
        record = self._require_active_record(session_id)
        expected = record.input.expected_bytes_per_frame
        if len(frame) != expected:
            record.stats.dropped_frames += 1
            raise TalkManagerError(
                f"Invalid PCM frame length: expected {expected} bytes, got {len(frame)}",
                reason=TalkRefusalReason.INVALID_AUDIO_FRAME,
            )
        assert record.session is not None
        await record.session.write_pcm_frame(frame)
        now = time.monotonic()
        record.last_activity_at = now
        record.stats.last_frame_time = now
        record.stats.frames_sent += 1
        record.stats.bytes_sent += len(frame)

    async def stop_session(self, session_id: str, *, reason: str = "client_stop") -> bool:
        """Stop a reserved or active talk session if it matches the current one."""
        self._bind_loop()
        record: _SessionRecord | None
        async with self._lock:
            record = self._record
            if record is None or record.session_id != session_id:
                return False
            self._clear_record_locked(record, close_reason=reason)
        await _close_record(record)
        return True

    async def shutdown(self) -> None:
        """Close any current talk session during source/runtime shutdown."""
        self._bind_loop()
        record: _SessionRecord | None
        async with self._lock:
            record = self._record
            if record is None:
                return
            self._clear_record_locked(record, close_reason="shutdown")
        await _close_record(record)

    def shutdown_sync(self, *, timeout_s: float) -> None:
        """Synchronously close the manager from a source thread."""
        loop = self._loop
        running_loop: asyncio.AbstractEventLoop | None
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if loop is not None and loop.is_running():
            if loop is running_loop:
                raise RuntimeError("Cannot synchronously stop talk manager from its event loop")
            future = asyncio.run_coroutine_threadsafe(self.shutdown(), loop)
            try:
                future.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise
            return

        if running_loop is not None:
            raise RuntimeError("Cannot synchronously stop talk manager from a running event loop")
        asyncio.run(self.shutdown())

    def _bind_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
            return
        if self._loop is not loop:
            raise TalkManagerError(
                "Talk manager cannot be used from multiple event loops",
                reason=TalkRefusalReason.RUNTIME_UNAVAILABLE,
            )

    def _require_active_record(self, session_id: str) -> _SessionRecord:
        record = self._record
        if record is None or record.session_id != session_id or record.session is None:
            raise TalkManagerError(
                "Talk session is not active",
                reason=TalkRefusalReason.RUNTIME_UNAVAILABLE,
            )
        return record

    async def _max_duration_watch(self, session_id: str) -> None:
        await asyncio.sleep(self._max_session_s)
        await self.stop_session(session_id, reason="max_duration_reached")

    async def _idle_watch(self, session_id: str) -> None:
        while True:
            await asyncio.sleep(self._idle_timeout_s)
            record = self._record
            if record is None or record.session_id != session_id:
                return
            if time.monotonic() - record.last_activity_at >= self._idle_timeout_s:
                await self.stop_session(session_id, reason="idle_timeout")
                return

    async def _clear_opening_record(self, session_id: str, *, close_reason: str) -> None:
        async with self._lock:
            record = self._record
            if record is None or record.session_id != session_id:
                return
            record.opening = False
            self._clear_record_locked(record, close_reason=close_reason)

    def _clear_record_locked(self, record: _SessionRecord, *, close_reason: str) -> None:
        if self._record is record:
            self._record = None
        record.state = TalkState.STOPPING
        record.stats.close_reason = close_reason
        for task in (record.timeout_task, record.idle_task):
            if task is not None and task is not asyncio.current_task():
                task.cancel()


class TalkManagerError(RuntimeError):
    """Raised when a talk manager operation is refused."""

    def __init__(self, message: str, *, reason: TalkRefusalReason) -> None:
        super().__init__(message)
        self.reason = reason


async def _close_record(record: _SessionRecord) -> None:
    session = record.session
    if session is None:
        return
    await session.close()


async def _close_session_safely(session: TalkSession) -> None:
    try:
        await session.close()
    except Exception as exc:  # pragma: no cover - defensive orphan cleanup
        logger.warning("Failed to close unclaimed talk session: %s", exc, exc_info=True)


def _new_session_id() -> str:
    return "tk_" + secrets.token_urlsafe(16)
