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
    TalkCapabilityProbeResult,
    TalkCapabilityState,
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
    io_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    timeout_task: asyncio.Task[None] | None = None
    idle_task: asyncio.Task[None] | None = None


class TalkManager:
    """Enforce one active/reserved talk session for a camera and clean it up."""

    def __init__(
        self,
        *,
        camera_name: str,
        enabled: bool,
        policy_enabled: bool | None = None,
        supported_codecs: list[str],
        open_session_factory: Callable[[TalkSessionOpenRequest], Awaitable[TalkSession]],
        max_session_s: float,
        idle_timeout_s: float,
        capability_probe_factory: Callable[[], Awaitable[TalkCapabilityProbeResult]] | None = None,
        capability_ttl_s: float = 30.0,
    ) -> None:
        self._camera_name = camera_name
        self._enabled = enabled
        self._policy_enabled = enabled if policy_enabled is None else policy_enabled
        self._supported_codecs = list(supported_codecs)
        self._open_session_factory = open_session_factory
        self._capability_probe_factory = capability_probe_factory
        self._capability_ttl_s = capability_ttl_s
        self._capability = TalkCapabilityProbeResult(
            capability=(
                TalkCapabilityState.DISABLED
                if not enabled
                else (
                    TalkCapabilityState.SUPPORTED
                    if capability_probe_factory is None
                    else TalkCapabilityState.UNKNOWN
                )
            )
        )
        self._capability_refreshed_at: float | None = None
        self._max_session_s = max_session_s
        self._idle_timeout_s = idle_timeout_s
        self._record: _SessionRecord | None = None
        self._lock = asyncio.Lock()
        self._probe_lock = asyncio.Lock()
        self._last_error: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def status(self) -> CameraTalkStatus:
        """Return current source-level talk status."""
        record = self._record
        capability = self._capability
        if not self._enabled:
            state = TalkState.DISABLED
        elif record is None:
            state = _state_from_capability(capability.capability)
        else:
            state = record.state
        return CameraTalkStatus(
            camera_name=self._camera_name,
            enabled=self._enabled,
            policy_enabled=self._policy_enabled,
            capability=capability.capability,
            state=state,
            active_session_id=record.session_id if record is not None else None,
            supported_codecs=self._supported_codecs,
            offered_codecs=capability.offered_codecs,
            selected_codec=(
                record.selected_codec if record is not None else capability.selected_codec
            ),
            last_error=self._last_error or capability.message,
        )

    async def refresh_status(self, *, force: bool = False) -> CameraTalkStatus:
        """Refresh capability if needed and return current source-level status."""
        await self.refresh_capability(force=force)
        return self.status()

    async def refresh_capability(
        self,
        *,
        force: bool = False,
    ) -> TalkCapabilityProbeResult:
        """Probe and cache camera talk capability when this manager has a probe source."""
        self._bind_loop()
        if not self._enabled:
            return self._capability
        if self._capability_probe_factory is None:
            return self._capability

        async with self._probe_lock:
            async with self._lock:
                if self._record is not None:
                    return self._capability
                now = time.monotonic()
                if not force and self._capability_is_fresh(now):
                    return self._capability
                self._capability = TalkCapabilityProbeResult(
                    capability=TalkCapabilityState.PROBING,
                    offered_codecs=self._capability.offered_codecs,
                    selected_codec=self._capability.selected_codec,
                    message=self._capability.message,
                    refusal_reason=self._capability.refusal_reason,
                )

            try:
                result = await self._capability_probe_factory()
            except Exception as exc:
                result = TalkCapabilityProbeResult(
                    capability=TalkCapabilityState.ERROR,
                    refusal_reason=TalkRefusalReason.CAMERA_BACKCHANNEL_FAILED,
                    message=str(exc) or type(exc).__name__,
                )

            async with self._lock:
                if self._record is None:
                    self._capability = result
                    self._capability_refreshed_at = time.monotonic()
                    if result.capability == TalkCapabilityState.SUPPORTED:
                        self._last_error = None
                return self._capability

    async def prepare_session(
        self,
        request: TalkSessionPrepareRequest,
    ) -> TalkSessionPrepareResult:
        """Reserve a talk slot before the browser attaches the WebSocket stream."""
        self._bind_loop()
        capability = await self.refresh_capability(force=True)
        if capability.capability != TalkCapabilityState.SUPPORTED:
            return _prepare_refusal_from_capability(capability, input_format=request.input)

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

    def _capability_is_fresh(self, now: float) -> bool:
        return (
            self._capability_refreshed_at is not None
            and now - self._capability_refreshed_at < self._capability_ttl_s
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
                if (
                    record.timeout_task is not None
                    and record.timeout_task is not asyncio.current_task()
                ):
                    record.timeout_task.cancel()
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
        async with self._lock:
            record = self._require_active_record(session_id)
            expected = record.input.expected_bytes_per_frame
            if len(frame) != expected:
                record.stats.dropped_frames += 1
                raise TalkManagerError(
                    f"Invalid PCM frame length: expected {expected} bytes, got {len(frame)}",
                    reason=TalkRefusalReason.INVALID_AUDIO_FRAME,
                )

        async with record.io_lock:
            async with self._lock:
                if self._record is not record or record.session is None:
                    raise TalkManagerError(
                        "Talk session is not active",
                        reason=TalkRefusalReason.RUNTIME_UNAVAILABLE,
                    )
                session = record.session

            await session.write_pcm_frame(frame)

            now = time.monotonic()
            async with self._lock:
                if self._record is record and record.session is session:
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


def _state_from_capability(capability: TalkCapabilityState) -> TalkState:
    match capability:
        case TalkCapabilityState.DISABLED:
            return TalkState.DISABLED
        case TalkCapabilityState.SUPPORTED:
            return TalkState.IDLE
        case TalkCapabilityState.UNSUPPORTED | TalkCapabilityState.UNSUPPORTED_CODEC:
            return TalkState.UNSUPPORTED
        case TalkCapabilityState.ERROR:
            return TalkState.ERROR
        case TalkCapabilityState.UNKNOWN | TalkCapabilityState.PROBING:
            return TalkState.TEMPORARILY_UNAVAILABLE


def _prepare_refusal_from_capability(
    capability: TalkCapabilityProbeResult,
    *,
    input_format: TalkInputFormat,
) -> TalkSessionPrepareResult:
    reason = capability.refusal_reason or _refusal_reason_from_capability(capability.capability)
    message = capability.message or _refusal_message_from_capability(capability.capability)
    return TalkSessionPrepareResult(
        accepted=False,
        refusal_reason=reason,
        message=message,
        input=input_format,
    )


def _refusal_reason_from_capability(
    capability: TalkCapabilityState,
) -> TalkRefusalReason:
    match capability:
        case TalkCapabilityState.DISABLED:
            return TalkRefusalReason.TALK_DISABLED
        case TalkCapabilityState.UNSUPPORTED:
            return TalkRefusalReason.UNSUPPORTED_CAMERA
        case TalkCapabilityState.UNSUPPORTED_CODEC:
            return TalkRefusalReason.UNSUPPORTED_CODEC
        case TalkCapabilityState.ERROR:
            return TalkRefusalReason.CAMERA_BACKCHANNEL_FAILED
        case (
            TalkCapabilityState.UNKNOWN
            | TalkCapabilityState.PROBING
            | TalkCapabilityState.SUPPORTED
        ):
            return TalkRefusalReason.RUNTIME_UNAVAILABLE


def _refusal_message_from_capability(capability: TalkCapabilityState) -> str:
    match capability:
        case TalkCapabilityState.DISABLED:
            return "Talk is disabled for this camera"
        case TalkCapabilityState.UNSUPPORTED:
            return "Camera does not advertise an ONVIF talk backchannel"
        case TalkCapabilityState.UNSUPPORTED_CODEC:
            return "Camera talk backchannel codec is not supported"
        case TalkCapabilityState.ERROR:
            return "Camera talk backchannel probe failed"
        case TalkCapabilityState.UNKNOWN | TalkCapabilityState.PROBING:
            return "Camera talk capability is not ready"
        case TalkCapabilityState.SUPPORTED:
            return "Talk session is temporarily unavailable"


async def _close_record(record: _SessionRecord) -> None:
    async with record.io_lock:
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
