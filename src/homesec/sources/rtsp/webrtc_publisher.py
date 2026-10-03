"""Source-owned supervision for the Rust WebRTC preview helper.

Only typed signaling crosses the runtime boundary. FFmpeg sends RTP directly to
helper loopback sockets; media and camera credentials never enter logs or files.
"""

from __future__ import annotations

import logging
import os
import selectors
import signal
import subprocess
import time
import uuid
from queue import Empty, Queue
from threading import Event, Lock, RLock, Thread
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from homesec.models.config import WebRTCPreviewConfig
from homesec.models.preview import (
    PreviewAnswer,
    PreviewOffer,
    PreviewSessionAction,
    PreviewSessionRefusal,
    PreviewSessionRefusalReason,
)
from homesec.sources.rtsp.capabilities import RTSPTimeoutCapabilities
from homesec.sources.rtsp.live_publisher import (
    LivePublisherRefusalReason,
    LivePublisherStartRefusal,
    LivePublisherState,
    LivePublisherStatus,
)

logger = logging.getLogger(__name__)
_MAX_MESSAGE_BYTES = 128_000


@runtime_checkable
class WebRTCPreviewPublisher(Protocol):
    def negotiate(
        self, offer: PreviewOffer, lease_expires_at: float
    ) -> PreviewAnswer | PreviewSessionRefusal: ...

    def renew(
        self, session_id: str, lease_expires_at: float
    ) -> PreviewSessionAction | PreviewSessionRefusal: ...

    def close(self, session_id: str) -> PreviewSessionAction | PreviewSessionRefusal: ...


class _HelperMessage(BaseModel):
    request_id: str | None = None
    event: str | None = None
    ok: bool = False
    error_code: str | None = None
    sdp: str | None = Field(default=None, max_length=48_000)
    viewer_count: int = Field(default=0, ge=0, le=32)
    media_active: bool = True
    state: Literal["starting", "ready", "error"] | None = None
    video_port: int | None = Field(default=None, ge=1, le=65535)
    audio_port: int | None = Field(default=None, ge=1, le=65535)
    media_port: int | None = Field(default=None, ge=1, le=65535)


class _HelperRequest(BaseModel):
    model_config = {"extra": "forbid"}

    request_id: str
    command: Literal["start", "offer", "renew", "close", "status", "stop"]
    ffmpeg_args: list[str] | None = None
    session_id: str | None = None
    sdp: str | None = Field(default=None, max_length=48_000)
    lease_seconds: float | None = Field(default=None, gt=0.0, le=86400.0)
    lease_expires_at: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def require_command_fields(self) -> _HelperRequest:
        if self.command == "start" and not self.ffmpeg_args:
            raise ValueError("Media startup requires arguments")
        if self.command in ("offer", "renew", "close") and not self.session_id:
            raise ValueError("Peer commands require a session")
        if self.command == "offer" and not self.sdp:
            raise ValueError("Negotiation requires an offer")
        if self.command in ("offer", "renew") and self.lease_seconds is None:
            raise ValueError("Peer authorization requires a lease")
        return self


class _HelperError(RuntimeError):
    """Bounded transport failure; messages deliberately contain no input data."""


class _HelperClient:
    def __init__(self, args: list[str]) -> None:
        self.process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            bufsize=0,
        )
        if self.process.stdin is not None:
            os.set_blocking(self.process.stdin.fileno(), False)
        self._pending: dict[str, Queue[_HelperMessage | None]] = {}
        self._pending_lock = Lock()
        self._write_lock = Lock()
        self._ready: Queue[_HelperMessage | None] = Queue(maxsize=1)
        self._reader = Thread(target=self._read, name="webrtc-helper-replies", daemon=True)
        self._reader.start()

    def wait_ready(self, timeout_s: float) -> _HelperMessage:
        try:
            result = self._ready.get(timeout=timeout_s)
        except Empty as exc:
            raise _HelperError("Preview helper startup timed out") from exc
        if (
            result is None
            or result.event != "ready"
            or result.video_port is None
            or result.audio_port is None
        ):
            raise _HelperError("Preview helper did not become ready")
        return result

    def request(self, command: str, *, timeout_s: float = 2.0, **fields: object) -> _HelperMessage:
        deadline = time.monotonic() + timeout_s
        request_id = str(uuid.uuid4())
        waiter: Queue[_HelperMessage | None] = Queue(maxsize=1)
        with self._pending_lock:
            if self.process.poll() is not None:
                raise _HelperError("Preview helper exited")
            if len(self._pending) >= 64:
                raise _HelperError("Preview helper command limit reached")
            self._pending[request_id] = waiter
        try:
            try:
                request = _HelperRequest.model_validate(
                    {"command": command, "request_id": request_id, **fields}
                )
            except ValueError as exc:
                raise _HelperError("Invalid preview helper command") from exc
            payload = request.model_dump_json(exclude_none=True).encode() + b"\n"
            if len(payload) > _MAX_MESSAGE_BYTES:
                raise _HelperError("Preview helper request exceeds limit")
            self._write(payload, deadline)
            try:
                result = waiter.get(timeout=max(0.0, deadline - time.monotonic()))
            except Empty as exc:
                raise _HelperError("Preview helper command timed out") from exc
            if result is None:
                raise _HelperError("Preview helper disconnected")
            return result
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

    def _write(self, payload: bytes, deadline: float) -> None:
        if not self._write_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise _HelperError("Preview helper command timed out")
        try:
            stdin = self.process.stdin
            if stdin is None:
                raise _HelperError("Preview helper input unavailable")
            with selectors.DefaultSelector() as selector:
                descriptor = stdin.fileno()
                selector.register(descriptor, selectors.EVENT_WRITE)
                remaining = memoryview(payload)
                while remaining:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        raise _HelperError("Preview helper command timed out")
                    try:
                        written = os.write(descriptor, remaining)
                    except BlockingIOError:
                        if not selector.select(timeout):
                            raise _HelperError("Preview helper command timed out") from None
                        continue
                    except InterruptedError:
                        continue
                    if written == 0:
                        raise _HelperError("Preview helper disconnected")
                    remaining = remaining[written:]
        except (OSError, ValueError) as exc:
            raise _HelperError("Preview helper disconnected") from exc
        finally:
            self._write_lock.release()

    def _read(self) -> None:
        stdout = self.process.stdout
        try:
            if stdout is None:
                return
            while True:
                raw = stdout.readline(_MAX_MESSAGE_BYTES + 1)
                if not raw:
                    return
                if len(raw) > _MAX_MESSAGE_BYTES or not raw.endswith(b"\n"):
                    return
                message = _HelperMessage.model_validate_json(raw)
                if message.event == "ready":
                    if self._ready.empty():
                        self._ready.put_nowait(message)
                elif message.request_id is not None:
                    with self._pending_lock:
                        waiter = self._pending.get(message.request_id)
                        if waiter is not None and waiter.empty():
                            waiter.put_nowait(message)
        except (OSError, ValueError):
            # Protocol failures can contain SDP or transport details: never log them.
            return
        finally:
            with self._pending_lock:
                for waiter in self._pending.values():
                    if waiter.empty():
                        waiter.put_nowait(None)
            if self._ready.empty():
                self._ready.put_nowait(None)

    def stop(self) -> None:
        """Terminate the helper and its FFmpeg child as one bounded process group."""
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=2.0)
        # A helper could have exited while leaving its FFmpeg child in the group.
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()
        self._reader.join(timeout=1.0)


class RustWebRTCLivePublisher:
    """One bounded helper and one RTSP reader per active camera preview."""

    def __init__(
        self,
        *,
        camera_name: str,
        rtsp_url: str,
        config: WebRTCPreviewConfig,
        idle_timeout_s: float,
        recording_policy: Literal["stop_on_recording", "allow_during_recording"],
        rtsp_connect_timeout_s: float,
        rtsp_io_timeout_s: float,
        timeout_capabilities: RTSPTimeoutCapabilities,
    ) -> None:
        self._camera_name = camera_name
        self._rtsp_url = rtsp_url
        self._config = config
        self._audio_available = False
        self._idle_timeout_s = idle_timeout_s
        self._recording_policy = recording_policy
        self._connect_timeout_s = rtsp_connect_timeout_s
        self._io_timeout_s = rtsp_io_timeout_s
        self._timeout_capabilities = timeout_capabilities
        self._lock = RLock()
        self._startup_lock = Lock()
        self._helper: _HelperClient | None = None
        self._generation = 0
        self._recording_active = False
        self._degraded_reason: str | None = None
        self._status = LivePublisherStatus(state=LivePublisherState.IDLE, viewer_count=0)
        self._last_viewer_at = time.monotonic()
        self._media_start_at = self._last_viewer_at
        self._shutdown = Event()
        self._maintenance = Thread(
            target=self._maintain, name="webrtc-preview-maintenance", daemon=True
        )
        self._maintenance.start()

    def status(self) -> LivePublisherStatus:
        with self._lock:
            return self._status

    def set_audio_available(self, available: bool) -> None:
        """Reuse startup discovery rather than opening another camera probe session."""
        self._audio_available = available

    def ensure_active(self) -> LivePublisherStatus | LivePublisherStartRefusal:
        with self._lock:
            generation = self._generation
        with self._startup_lock:
            with self._lock:
                if self._recording_blocks():
                    return self._recording_refusal()
                if generation != self._generation or self._shutdown.is_set():
                    return self._unavailable()
                if self._helper is not None and self._helper.process.poll() is None:
                    return self._status
                self._status = LivePublisherStatus(
                    state=LivePublisherState.STARTING, viewer_count=0
                )
                self._media_start_at = time.monotonic()
            helper: _HelperClient | None = None
            try:
                helper = _HelperClient(
                    [
                        self._config.helper_path,
                        "--advertised-ip",
                        self._config.advertised_ip,
                        "--udp-port-start",
                        str(self._config.udp_port_start),
                        "--udp-port-end",
                        str(self._config.udp_port_end),
                        "--max-viewers",
                        str(self._config.max_viewers),
                        "--negotiation-timeout-s",
                        str(self._config.negotiation_timeout_s),
                        "--max-session-duration-s",
                        str(self._config.max_session_duration_s),
                    ]
                )
                with self._lock:
                    if generation != self._generation or self._recording_blocks():
                        helper.stop()
                        return (
                            self._recording_refusal()
                            if self._recording_blocks()
                            else self._unavailable()
                        )
                    self._helper = helper
                ready = helper.wait_ready(timeout_s=5.0)
                response = helper.request("start", ffmpeg_args=self._ffmpeg_args(ready))
                if not response.ok:
                    raise _HelperError("Preview media startup failed")
                with self._lock:
                    if self._helper is not helper or generation != self._generation:
                        return (
                            self._recording_refusal()
                            if self._recording_blocks()
                            else self._unavailable()
                        )
                    self._last_viewer_at = time.monotonic()
                    self._status = self._running_status(response.viewer_count)
                    return self._status
            except (OSError, _HelperError) as exc:
                if helper is not None:
                    helper.stop()
                with self._lock:
                    if generation != self._generation:
                        return (
                            self._recording_refusal()
                            if self._recording_blocks()
                            else self._unavailable()
                        )
                    self._helper = None
                    self._status = LivePublisherStatus(
                        state=LivePublisherState.ERROR,
                        viewer_count=0,
                        last_error="WebRTC preview helper is unavailable",
                    )
                logger.warning(
                    "WebRTC preview startup failed",
                    extra={
                        "camera_name": self._camera_name,
                        "event_type": "preview_start_failed",
                        "error_type": type(exc).__name__,
                    },
                )
                return self._unavailable()

    def negotiate(
        self, offer: PreviewOffer, lease_expires_at: float
    ) -> PreviewAnswer | PreviewSessionRefusal:
        if lease_expires_at <= time.time():
            return self._session_unavailable()
        outcome = self.ensure_active()
        if isinstance(outcome, LivePublisherStartRefusal):
            return PreviewSessionRefusal(
                reason=PreviewSessionRefusalReason(outcome.reason.value), message=outcome.message
            )
        if lease_expires_at <= time.time():
            return self._session_unavailable()
        session_id = str(uuid.uuid4())
        response = self._session_request(
            "offer",
            session_id=session_id,
            sdp=offer.sdp,
            lease_expires_at=lease_expires_at,
            lease_seconds=min(lease_expires_at - time.time(), self._config.max_session_duration_s),
            timeout_s=self._config.negotiation_timeout_s + 2.0,
        )
        if isinstance(response, PreviewSessionRefusal):
            return response
        if response.sdp is None:
            self.close(session_id)
            return self._session_unavailable()
        return PreviewAnswer(session_id=session_id, sdp=response.sdp)

    def renew(
        self, session_id: str, lease_expires_at: float
    ) -> PreviewSessionAction | PreviewSessionRefusal:
        remaining_s = lease_expires_at - time.time()
        if remaining_s <= 0:
            return self._session_unavailable()
        response = self._session_request(
            "renew",
            session_id=session_id,
            lease_expires_at=lease_expires_at,
            lease_seconds=min(remaining_s, self._config.max_session_duration_s),
        )
        if isinstance(response, PreviewSessionRefusal):
            return response
        return PreviewSessionAction(accepted=True)

    def close(self, session_id: str) -> PreviewSessionAction | PreviewSessionRefusal:
        with self._lock:
            if self._helper is None:
                return PreviewSessionAction(accepted=True)
        response = self._session_request("close", session_id=session_id)
        if isinstance(response, PreviewSessionRefusal):
            return response
        return PreviewSessionAction(accepted=True)

    def _session_request(
        self, command: str, *, timeout_s: float = 2.0, **fields: object
    ) -> _HelperMessage | PreviewSessionRefusal:
        with self._lock:
            helper = self._helper
            if command != "close" and self._recording_blocks():
                return self._session_recording_refusal()
            if command != "close" and self._shutdown.is_set():
                return self._session_unavailable()
        if helper is None:
            return self._session_unavailable()
        response: _HelperMessage | None
        try:
            response = helper.request(command, timeout_s=timeout_s, **fields)
        except _HelperError:
            # Negotiation timeout must not leave an untracked viewer behind.
            if command == "offer":
                try:
                    helper.request("close", session_id=fields.get("session_id"))
                except _HelperError:
                    self._stop_owned_helper(
                        expected=helper, error="WebRTC preview helper disconnected"
                    )
            response = None
        with self._lock:
            if command != "close" and self._recording_blocks():
                return self._session_recording_refusal()
            if command != "close" and (self._helper is not helper or self._shutdown.is_set()):
                return self._session_unavailable()
        if response is None:
            return self._session_unavailable()
        if not response.ok:
            try:
                reason = PreviewSessionRefusalReason(
                    response.error_code or "preview_temporarily_unavailable"
                )
            except ValueError:
                reason = PreviewSessionRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE
            return PreviewSessionRefusal(reason=reason, message="Preview session was refused")
        return response

    def downgrade_concurrent_preview(self, reason: str) -> None:
        with self._lock:
            self._degraded_reason = reason
            recording = self._recording_active
        if recording:
            self.request_stop()

    def sync_recording_active(self, recording_active: bool) -> None:
        with self._lock:
            self._recording_active = recording_active
            stop = self._recording_blocks()
        if stop:
            self.request_stop()

    def note_viewer_activity(self, viewer_id: str | None = None) -> None:
        # HTTP polling must never extend WebRTC peer leases or input lifetime.
        _ = viewer_id

    def request_stop(self) -> None:
        self._stop_owned_helper()

    def _stop_owned_helper(
        self, *, expected: _HelperClient | None = None, error: str | None = None
    ) -> None:
        with self._lock:
            if expected is not None and self._helper is not expected:
                return
            self._generation += 1
            generation = self._generation
            helper = self._helper
            self._helper = None
            self._status = LivePublisherStatus(state=LivePublisherState.STOPPING, viewer_count=0)
        if helper is not None:
            helper.stop()
        with self._lock:
            # An old teardown must never overwrite the state of a newer activation.
            if self._generation == generation and self._helper is None:
                self._status = LivePublisherStatus(
                    state=LivePublisherState.ERROR if error else LivePublisherState.IDLE,
                    viewer_count=0,
                    degraded_reason=self._degraded_reason,
                    last_error=error,
                )

    def shutdown(self) -> None:
        self._shutdown.set()
        self.request_stop()
        self._maintenance.join(timeout=3.0)

    def _maintain(self) -> None:
        while not self._shutdown.wait(0.5):
            with self._lock:
                helper = self._helper
                ready = self._status.state in (
                    LivePublisherState.STARTING,
                    LivePublisherState.READY,
                    LivePublisherState.DEGRADED,
                )
            if helper is None or not ready:
                continue
            try:
                response = helper.request("status")
            except _HelperError:
                self._stop_owned_helper(expected=helper, error="WebRTC preview helper exited")
                continue
            if (
                response.ok
                and response.state == "starting"
                and time.monotonic() - self._media_start_at < 10.0
            ):
                with self._lock:
                    if self._helper is helper:
                        self._status = LivePublisherStatus(
                            state=LivePublisherState.STARTING, viewer_count=response.viewer_count
                        )
                continue
            if not response.ok or not response.media_active:
                self._stop_owned_helper(
                    expected=helper, error="WebRTC preview media is unavailable"
                )
                continue
            with self._lock:
                if self._helper is not helper:
                    continue
                if response.viewer_count:
                    self._last_viewer_at = time.monotonic()
                idle = (
                    not response.viewer_count
                    and time.monotonic() - self._last_viewer_at >= self._idle_timeout_s
                )
                self._status = self._running_status(response.viewer_count)
            if idle:
                self._stop_owned_helper(expected=helper)

    def _running_status(self, viewers: int) -> LivePublisherStatus:
        return LivePublisherStatus(
            state=LivePublisherState.DEGRADED
            if self._degraded_reason
            else LivePublisherState.READY,
            viewer_count=viewers,
            degraded_reason=self._degraded_reason,
            idle_shutdown_at=(self._last_viewer_at + self._idle_timeout_s if not viewers else None),
        )

    def _recording_blocks(self) -> bool:
        return self._recording_active and (
            self._recording_policy == "stop_on_recording" or self._degraded_reason is not None
        )

    @staticmethod
    def _recording_refusal() -> LivePublisherStartRefusal:
        return LivePublisherStartRefusal(
            reason=LivePublisherRefusalReason.RECORDING_PRIORITY,
            message="Preview yields to active recording",
        )

    @staticmethod
    def _unavailable() -> LivePublisherStartRefusal:
        return LivePublisherStartRefusal(
            reason=LivePublisherRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE,
            message="WebRTC preview is unavailable",
        )

    @staticmethod
    def _session_recording_refusal() -> PreviewSessionRefusal:
        return PreviewSessionRefusal(
            reason=PreviewSessionRefusalReason.RECORDING_PRIORITY,
            message="Preview yields to active recording",
        )

    @staticmethod
    def _session_unavailable() -> PreviewSessionRefusal:
        return PreviewSessionRefusal(
            reason=PreviewSessionRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE,
            message="WebRTC preview is unavailable",
        )

    def _ffmpeg_args(self, ready: _HelperMessage) -> list[str]:
        args = ["-hide_banner", "-nostdin", "-loglevel", "error", "-rtsp_transport", "tcp"]
        args.extend(
            self._timeout_capabilities.build_ffmpeg_timeout_args(
                connect_timeout_s=self._connect_timeout_s, io_timeout_s=self._io_timeout_s
            )
        )
        args.extend(["-fflags", "+genpts+igndts", "-i", self._rtsp_url, "-map", "0:v:0", "-an"])
        if self._config.video_codec == "copy":
            args.extend(["-c:v", "copy"])
        else:
            args.extend(
                [
                    "-c:v",
                    "libx264",
                    "-profile:v",
                    "baseline",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    "veryfast",
                    "-tune",
                    "zerolatency",
                    "-bf",
                    "0",
                    "-sc_threshold",
                    "0",
                    "-force_key_frames",
                    "expr:gte(t,n_forced*1)",
                ]
            )
        args.extend(
            [
                "-f",
                "rtp",
                "-payload_type",
                "96",
                f"rtp://127.0.0.1:{ready.video_port}?pkt_size=1200",
            ]
        )
        if self._config.audio_enabled and self._audio_available:
            args.extend(
                [
                    "-map",
                    "0:a:0?",
                    "-vn",
                    "-c:a",
                    "libopus",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-b:a",
                    "64k",
                    "-application",
                    "lowdelay",
                    "-frame_duration",
                    "20",
                    "-f",
                    "rtp",
                    "-payload_type",
                    "97",
                    f"rtp://127.0.0.1:{ready.audio_port}?pkt_size=1200",
                ]
            )
        return args
