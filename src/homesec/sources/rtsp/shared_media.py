"""Source-owned consumers of one bounded Rust media helper.

Only control, observations, and completed file paths cross the Python boundary.
Consumer teardown detaches that consumer; the source owns process teardown.
"""

from __future__ import annotations

import logging
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from threading import RLock
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from homesec.models.config import WebRTCPreviewConfig
from homesec.sources.rtsp.helper_client import HelperClient, HelperError
from homesec.sources.rtsp.motion_input import MotionObservation
from homesec.sources.rtsp.recorder import FfmpegRecorder, RecordingHandle
from homesec.sources.rtsp.recording_profile import (
    RecordingProfile,
    build_recording_profile_candidates,
)
from homesec.sources.rtsp.rust_motion import _MotionReply, _MotionSettings
from homesec.sources.rtsp.webrtc_publisher import (
    _HelperClient,
    _HelperMessage,
    _PreviewHelper,
)

logger = logging.getLogger(__name__)


class _SharedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    request_id: str = Field(max_length=128)
    command: Literal[
        "start",
        "offer",
        "renew",
        "close",
        "status",
        "stop_preview",
        "stop",
        "start_motion",
        "read_motion",
        "discard_frame",
        "stop_motion",
        "start_recording",
        "stop_recording",
        "recording_status",
    ]
    rtsp_url: str | None = Field(default=None, min_length=1, max_length=16 * 1024)
    session_id: str | None = Field(default=None, max_length=128)
    sdp: str | None = Field(default=None, max_length=48_000)
    lease_seconds: float | None = Field(default=None, gt=0, le=86400)
    lease_expires_at: float | None = Field(default=None, gt=0)
    motion_config: _MotionSettings | None = None
    motion_id: str | None = Field(default=None, min_length=1, max_length=128)
    frame_queue_size: int | None = Field(default=None, ge=1)
    connect_timeout_s: float | None = Field(default=None, gt=0, le=120)
    io_timeout_s: float | None = Field(default=None, gt=0, le=120)
    wait_timeout_s: float | None = Field(default=None, ge=0, le=120)
    threshold: float | None = Field(default=None, ge=0)
    recording_id: str | None = Field(default=None, min_length=1, max_length=128)
    output_path: str | None = Field(default=None, min_length=1, max_length=16 * 1024)
    audio_mode: Literal["copy", "none"] | None = None

    @model_validator(mode="after")
    def require_fields(self) -> _SharedRequest:
        if self.command in ("start", "start_motion", "start_recording") and not self.rtsp_url:
            raise ValueError("Media startup requires an input")
        if self.command == "start_motion" and any(
            value is None
            for value in (
                self.motion_config,
                self.frame_queue_size,
                self.connect_timeout_s,
                self.io_timeout_s,
            )
        ):
            raise ValueError("Motion startup requires validated settings")
        if self.command in ("read_motion", "discard_frame") and self.wait_timeout_s is None:
            raise ValueError("Frame consumption requires a deadline")
        if self.command == "read_motion" and self.threshold is None:
            raise ValueError("Motion consumption requires a threshold")
        if self.command in ("start_recording", "stop_recording", "recording_status"):
            if self.recording_id is None:
                raise ValueError("Recording commands require an owner")
        if self.command == "start_recording" and any(
            value is None
            for value in (
                self.output_path,
                self.audio_mode,
                self.connect_timeout_s,
                self.io_timeout_s,
            )
        ):
            raise ValueError("Recording startup requires an output and selected profile")
        if self.command in ("offer", "renew", "close") and not self.session_id:
            raise ValueError("Peer commands require a session")
        if self.command == "offer" and not self.sdp:
            raise ValueError("Negotiation requires an offer")
        if self.command in ("offer", "renew") and self.lease_seconds is None:
            raise ValueError("Peer authorization requires a lease")
        return self


class _SharedReply(_HelperMessage):
    observation: MotionObservation | None = None
    frame_available: bool = False
    recording_active: bool | None = None
    recording_finalized: bool = False


class SharedMediaSession:
    """One lazy helper per source, with independently detachable consumers."""

    def __init__(
        self,
        *,
        helper_path: str,
        preview_config: WebRTCPreviewConfig | None,
        connect_timeout_s: float = 10.0,
        io_timeout_s: float = 10.0,
    ) -> None:
        self._connect_timeout_s = connect_timeout_s
        self._io_timeout_s = io_timeout_s
        self._args = [helper_path, "--shared"]
        if preview_config is not None and preview_config.advertised_ip is not None:
            self._args.extend(
                [
                    "--advertised-ip",
                    preview_config.advertised_ip,
                    "--udp-port-start",
                    str(preview_config.udp_port_start),
                    "--udp-port-end",
                    str(preview_config.udp_port_end),
                    "--max-viewers",
                    str(preview_config.max_viewers),
                    "--negotiation-timeout-s",
                    str(preview_config.negotiation_timeout_s),
                    "--max-session-duration-s",
                    str(preview_config.max_session_duration_s),
                ]
            )
        self._lock = RLock()
        self._motion_lock = RLock()
        self._preview_lock = RLock()
        self._helper: HelperClient[_SharedRequest, _SharedReply] | None = None
        self._ready: _SharedReply | None = None
        self._motion: _SharedMotionClient | None = None
        self._preview: _SharedPreviewClient | None = None
        self._closed = False
        self._retiring = False

    def _message(self, message: _SharedReply) -> None:
        if message.event != "frame":
            return
        with self._lock:
            if self._closed or self._retiring:
                return
            motion = self._motion
        if motion is not None and not motion.stopped:
            motion.callback(_MotionReply.model_validate(message.model_dump()))

    def _client(self) -> tuple[HelperClient[_SharedRequest, _SharedReply], _SharedReply]:
        expired: HelperClient[_SharedRequest, _SharedReply] | None = None
        with self._lock:
            if self._closed:
                raise HelperError("Media source has stopped")
            if self._retiring and self._helper is not None and self._helper.process.poll() is None:
                raise HelperError("Media owner teardown has not completed")
            if self._helper is not None and self._helper.process.poll() is not None:
                expired = self._helper
                self._helper = None
                self._ready = None
                self._retiring = False
                if self._motion is not None and self._motion.helper is expired:
                    self._motion.stopped = True
                    self._motion = None
        if expired is not None:
            expired.stop()
        with self._lock:
            if self._closed:
                raise HelperError("Media source has stopped")
            if self._retiring:
                raise HelperError("Media owner teardown has not completed")
            if self._helper is None:
                helper = HelperClient(
                    self._args,
                    request_model=_SharedRequest,
                    reply_model=_SharedReply,
                    on_message=self._message,
                    context="Media",
                )
                try:
                    ready = helper.wait_ready(timeout_s=5.0)
                except HelperError:
                    helper.stop()
                    raise
                self._helper = helper
                self._ready = ready
            assert self._ready is not None
            return self._helper, self._ready

    def _retire(self, helper: HelperClient[_SharedRequest, _SharedReply]) -> bool:
        """Resolve lost recording ownership by retiring its private process."""
        with self._lock:
            if self._helper is helper:
                self._retiring = True
        # Reader callbacks need the owner lock; process teardown must not hold it.
        try:
            helper.stop()
        except (OSError, HelperError, subprocess.TimeoutExpired):
            pass
        dead = helper.process.poll() is not None
        if dead:
            with self._lock:
                if self._helper is helper:
                    self._helper = None
                    self._ready = None
                    self._retiring = False
                if self._motion is not None and self._motion.helper is helper:
                    self._motion.stopped = True
                    self._motion = None
            with self._preview_lock:
                if self._preview is not None and self._preview.helper is helper:
                    self._preview.stopped = True
                    self._preview = None
        return dead

    def motion_client(self, callback: Callable[[_MotionReply], None]) -> _SharedMotionClient:
        with self._motion_lock:
            helper, ready = self._client()
            with self._lock:
                previous = self._motion
            if previous is not None:
                previous.stop()
            client = _SharedMotionClient(self, helper, ready, callback)
            with self._lock:
                self._motion = client
            return client

    def preview_client(self, args: list[str]) -> _PreviewHelper:
        try:
            helper, ready = self._client()
        except (OSError, HelperError):
            with self._lock:
                if self._closed or self._retiring:
                    raise HelperError("Media source is unavailable") from None
            # Older installations retain the established standalone preview behavior.
            return _HelperClient(args)
        with self._preview_lock:
            if self._preview is not None:
                self._preview.stop()
            client = _SharedPreviewClient(self, helper, ready)
            self._preview = client
            return client

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            helper = self._helper
        if helper is not None:
            if not self._retire(helper):
                raise HelperError("Media source teardown was not confirmed")


class _SharedMotionClient:
    def __init__(
        self,
        owner: SharedMediaSession,
        helper: HelperClient[_SharedRequest, _SharedReply],
        ready: _SharedReply,
        callback: Callable[[_MotionReply], None],
    ) -> None:
        self.owner = owner
        self.helper = helper
        self.ready = ready
        self.callback = callback
        self.motion_id = uuid.uuid4().hex
        self.stopped = False

    @property
    def process(self) -> subprocess.Popen[bytes]:
        return self.helper.process

    def wait_ready(self, timeout_s: float) -> _MotionReply:
        return _MotionReply.model_validate(self.ready.model_dump())

    def request(self, command: str, *, timeout_s: float = 2.0, **fields: object) -> _MotionReply:
        command = "start_motion" if command == "start" else command
        fields["motion_id"] = self.motion_id
        with self.owner._motion_lock:
            with self.owner._lock:
                if self.stopped or self.owner._motion is not self:
                    raise HelperError("Motion consumer has stopped")
            if command == "start_motion":
                response = self.helper.request(command, timeout_s=timeout_s, **fields)
                return _MotionReply.model_validate(response.model_dump())
        # Consuming a frame may wait on slow detection. Keep cancellation free
        # to detach it; the native owner ID rejects a delayed old dispatch.
        response = self.helper.request(command, timeout_s=timeout_s, **fields)
        return _MotionReply.model_validate(response.model_dump())

    def stop(self) -> None:
        with self.owner._motion_lock:
            with self.owner._lock:
                self.stopped = True
                if self.owner._motion is not self:
                    return
            try:
                response = self.helper.request(
                    "stop_motion", timeout_s=0.5, motion_id=self.motion_id
                )
            except HelperError:
                if self.helper.process.poll() is None:
                    # Retain the owner until detach or process death is confirmed.
                    # An ordinary motion failure must not kill an active recorder.
                    raise
            else:
                if not response.ok:
                    raise HelperError("Motion detach was not confirmed")
            with self.owner._lock:
                if self.owner._motion is self:
                    self.owner._motion = None


class _SharedPreviewClient:
    def __init__(
        self,
        owner: SharedMediaSession,
        helper: HelperClient[_SharedRequest, _SharedReply],
        ready: _SharedReply,
    ) -> None:
        self.owner = owner
        self.helper = helper
        self.ready = ready
        self.stopped = False

    @property
    def process(self) -> subprocess.Popen[bytes]:
        return self.helper.process

    def wait_ready(self, timeout_s: float) -> _HelperMessage:
        if self.ready.video_port is None or self.ready.audio_port is None:
            raise HelperError("Preview helper did not become ready")
        return self.ready

    def request(self, command: str, *, timeout_s: float = 2.0, **fields: object) -> _HelperMessage:
        # Dispatch and reply belong to the same preview lifetime as startup and
        # detach. Recording and motion use this transport independently of the lock.
        with self.owner._preview_lock:
            if self.stopped or self.owner._preview is not self:
                raise HelperError("Preview consumer has stopped")
            if command == "start":
                fields.setdefault("connect_timeout_s", self.owner._connect_timeout_s)
                fields.setdefault("io_timeout_s", self.owner._io_timeout_s)
            return self.helper.request(command, timeout_s=timeout_s, **fields)

    def stop(self) -> None:
        with self.owner._preview_lock:
            self.stopped = True
            if self.owner._preview is not self:
                return
            try:
                response = self.helper.request("stop_preview", timeout_s=0.5)
            except HelperError:
                if self.helper.process.poll() is not None:
                    self.owner._preview = None
                    return
                # Keep ownership so a successor cannot race an input whose stop
                # was never acknowledged. Recording keeps its healthy process.
                raise
            if not response.ok:
                raise HelperError("Preview detach was not confirmed")
            self.owner._preview = None


class _NativeRecordingHandle:
    def __init__(
        self,
        owner: SharedRecorder,
        helper: HelperClient[_SharedRequest, _SharedReply],
        recording_id: str,
    ) -> None:
        self.owner = owner
        self.helper = helper
        self.recording_id = recording_id
        self.exit_code: int | None = None
        self.finalized: bool | None = None

    @property
    def pid(self) -> int:
        return self.helper.process.pid

    @property
    def returncode(self) -> int | None:
        return self.exit_code if self.exit_code is not None else self.helper.process.poll()


class SharedRecorder:
    """Copy eligible streams in Rust; keep negotiated FFmpeg behavior otherwise."""

    def __init__(
        self,
        *,
        session: SharedMediaSession,
        fallback: FfmpegRecorder,
        connect_timeout_s: float,
        io_timeout_s: float,
    ) -> None:
        self._session = session
        self._fallback = fallback
        self._connect_timeout_s = connect_timeout_s
        self._io_timeout_s = io_timeout_s
        self._profile: RecordingProfile | None = None
        self._eligible = False
        self._native_failed = False
        self._codecs: tuple[str | None, str | None] | None = None
        self._uncertain: dict[str, _NativeRecordingHandle] = {}

    def configure_profile(
        self,
        profile: RecordingProfile,
        *,
        video_codec: str | None,
        audio_codec: str | None,
    ) -> None:
        self._fallback.configure_profile(profile)
        codecs = (video_codec, audio_codec)
        if self._profile != profile or self._codecs != codecs:
            self._native_failed = False
        self._codecs = codecs
        self._profile = profile
        canonical_profiles = build_recording_profile_candidates(
            input_url=profile.input_url,
            audio_codec=audio_codec,
        )
        self._eligible = (
            video_codec == "h264"
            and not profile.ffmpeg_input_args
            and profile.audio_mode in ("copy", "none")
            and (profile.audio_mode == "none" or audio_codec in (None, "aac"))
            and profile in canonical_profiles
        )

    def start(self, output_file: Path, stderr_log: Path) -> RecordingHandle | None:
        if not self._resolve_uncertain():
            return None
        profile = self._profile
        attempted_native = False
        if self._eligible and not self._native_failed and profile is not None:
            attempted_native = True
            helper: HelperClient[_SharedRequest, _SharedReply] | None = None
            recording_id = str(uuid.uuid4())
            try:
                helper, _ = self._session._client()
                response = helper.request(
                    "start_recording",
                    timeout_s=max(2.0, self._connect_timeout_s + self._io_timeout_s + 1.0),
                    recording_id=recording_id,
                    rtsp_url=profile.input_url,
                    output_path=str(output_file.absolute()),
                    audio_mode=profile.audio_mode,
                    connect_timeout_s=self._connect_timeout_s,
                    io_timeout_s=self._io_timeout_s,
                )
                if response.ok and response.recording_active is True:
                    return _NativeRecordingHandle(self, helper, recording_id)
            except (OSError, HelperError):
                pass
            self._native_failed = True
            if helper is not None:
                # A lost startup reply may still have allocated a writer.
                try:
                    cleanup = helper.request(
                        "stop_recording", recording_id=recording_id, timeout_s=2.0
                    )
                    if (
                        cleanup.recording_active is False
                        and cleanup.ok
                        and cleanup.recording_finalized
                    ):
                        completed = _NativeRecordingHandle(self, helper, recording_id)
                        completed.finalized = True
                        completed.exit_code = 0
                        return completed
                    if cleanup.recording_active is not False:
                        self._uncertain[recording_id] = _NativeRecordingHandle(
                            self, helper, recording_id
                        )
                        self._resolve_uncertain()
                        return None
                except HelperError:
                    if helper.process.poll() is None:
                        self._uncertain[recording_id] = _NativeRecordingHandle(
                            self, helper, recording_id
                        )
                        self._resolve_uncertain()
                        return None
        # Orderly helper teardown can publish a final clip despite a lost reply.
        # Preserve that clip for replay, including retries after native failure.
        if output_file.exists() or output_file.is_symlink():
            logger.warning(
                "Recording output already exists; retrying requires a new path",
                extra={"recording_id": output_file.name},
            )
            return None
        if attempted_native:
            logger.info(
                "Using compatible FFmpeg recording",
                extra={"event_type": "recording_backend_fallback", "reason": "native_unavailable"},
            )
        return self._fallback.start(output_file, stderr_log)

    def _resolve_uncertain(self) -> bool:
        for recording_id, handle in tuple(self._uncertain.items()):
            if handle.helper.process.poll() is None and not self._session._retire(handle.helper):
                return False
            handle.exit_code = 1
            handle.finalized = False
            self._uncertain.pop(recording_id, None)
        return True

    def stop(self, proc: RecordingHandle, output_file: Path | None) -> bool | None:
        if not isinstance(proc, _NativeRecordingHandle):
            self._fallback.stop(proc, output_file)
            return None
        if proc.owner is not self:
            raise TypeError("Recording handle belongs to another recorder")
        if proc.finalized is not None:
            return proc.finalized
        try:
            response = proc.helper.request(
                "stop_recording",
                recording_id=proc.recording_id,
                timeout_s=5.0,
            )
            closed = response.recording_active is False
            finalized = closed and response.ok and response.recording_finalized
        except HelperError:
            closed = proc.helper.process.poll() is not None
            finalized = False
        if not closed:
            self._uncertain[proc.recording_id] = proc
            closed = self._resolve_uncertain()
        proc.exit_code = 0 if finalized else 1
        if closed:
            proc.finalized = finalized
        if not finalized:
            self._native_failed = True
        return finalized

    def is_alive(self, proc: RecordingHandle) -> bool:
        if not isinstance(proc, _NativeRecordingHandle):
            return self._fallback.is_alive(proc)
        if proc.owner is not self:
            raise TypeError("Recording handle belongs to another recorder")
        if proc.returncode is not None:
            if proc.returncode != 0:
                self._native_failed = True
            return False
        try:
            response = proc.helper.request("recording_status", recording_id=proc.recording_id)
            active = response.ok and response.recording_active is True
        except HelperError:
            active = False
        if not active:
            proc.exit_code = 1
            self._native_failed = True
        return active
