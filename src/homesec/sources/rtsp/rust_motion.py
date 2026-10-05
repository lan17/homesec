"""Source-owned Rust decoding/motion, with the existing input as compatibility fallback."""

from __future__ import annotations

import logging
import math
import shutil
from collections.abc import Callable
from threading import RLock
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from homesec.sources.rtsp.helper_client import HelperClient, HelperError, HelperMessage
from homesec.sources.rtsp.motion_input import MotionInput, MotionObservation
from homesec.sources.rtsp.recording_profile import MotionProfile

logger = logging.getLogger(__name__)


class _MotionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    pixel_threshold: int = Field(ge=0, le=2**64 - 1)
    min_changed_pct: float = Field(ge=0)
    blur_kernel: int = Field(ge=0, le=2**31 - 1)
    recording_sensitivity_factor: float = Field(ge=1)


class _MotionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    request_id: str = Field(max_length=128)
    command: Literal["start", "read_motion", "discard_frame", "status", "stop"]
    rtsp_url: str | None = Field(default=None, min_length=1, max_length=16 * 1024)
    motion_config: _MotionSettings | None = None
    frame_queue_size: int | None = Field(default=None, ge=1)
    connect_timeout_s: float | None = Field(default=None, gt=0, le=120)
    io_timeout_s: float | None = Field(default=None, gt=0, le=120)
    wait_timeout_s: float | None = Field(default=None, ge=0, le=120)
    threshold: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_fields(self) -> _MotionRequest:
        if self.command == "start" and any(
            value is None
            for value in (
                self.rtsp_url,
                self.motion_config,
                self.frame_queue_size,
                self.connect_timeout_s,
                self.io_timeout_s,
            )
        ):
            raise ValueError("Motion startup requires input and validated settings")
        if self.command in ("read_motion", "discard_frame") and self.wait_timeout_s is None:
            raise ValueError("Frame consumption requires a deadline")
        if self.command == "read_motion" and self.threshold is None:
            raise ValueError("Motion consumption requires a threshold")
        return self


class _MotionReply(HelperMessage):
    observation: MotionObservation | None = None
    frame_available: bool = False


class RustMotionInput:
    def __init__(
        self,
        *,
        fallback: MotionInput,
        settings: _MotionSettings,
        helper_path: str,
        frame_queue_size: int,
        rtsp_connect_timeout_s: float,
        rtsp_io_timeout_s: float,
        on_frame: Callable[[], None],
    ) -> None:
        self._fallback = fallback
        self._settings = settings
        self._helper_path = helper_path
        self._queue_size = frame_queue_size
        self._connect_timeout = rtsp_connect_timeout_s
        self._io_timeout = rtsp_io_timeout_s
        self._on_frame = on_frame
        self._lock = RLock()
        self._helper: HelperClient[_MotionRequest, _MotionReply] | None = None
        self._url: str | None = None
        self._fallback_url: str | None = None
        self._compatible_profile = True
        self._legacy_active = False
        self._stopped = True
        self._generation = 0
        self._frame_generation: int | None = None

    def _message(self, reply: _MotionReply, generation: int) -> None:
        if reply.event == "frame" and generation == self._generation and not self._stopped:
            self._on_frame()

    def set_motion_profile(self, profile: MotionProfile) -> None:
        self._fallback.set_motion_profile(profile)
        # Negotiated workaround flags must retain their established FFmpeg semantics.
        self._compatible_profile = not profile.ffmpeg_input_args

    def start(self, rtsp_url: str) -> None:
        with self._lock:
            self.stop()
            self._stopped = False
            self._url = rtsp_url
            if not self._compatible_profile or self._fallback_url == rtsp_url:
                self._start_fallback("incompatible_input")
                return
            reason: str | None = None
            generation = self._generation
            try:
                self._helper = HelperClient(
                    [self._helper_path, "--motion"],
                    request_model=_MotionRequest,
                    reply_model=_MotionReply,
                    on_message=lambda reply: self._message(reply, generation),
                    context="Motion",
                )
                self._helper.wait_ready(timeout_s=2.0)
                reply = self._helper.request(
                    "start",
                    rtsp_url=rtsp_url,
                    motion_config=self._settings,
                    frame_queue_size=self._queue_size,
                    connect_timeout_s=self._connect_timeout,
                    io_timeout_s=self._io_timeout,
                )
                if not reply.ok:
                    reason = "motion_worker_refused"
            except (OSError, HelperError):
                reason = "motion_worker_unavailable"
            if reason is not None:
                # Leave the handled exception context before starting legacy input.
                self._start_fallback(reason)

    def _start_fallback(
        self,
        reason: str,
        expected_helper: HelperClient[_MotionRequest, _MotionReply] | None = None,
    ) -> None:
        with self._lock:
            if (
                (expected_helper is not None and expected_helper is not self._helper)
                or self._stopped
                or self._url is None
            ):
                return
            if self._helper is not None:
                self._helper.stop()
                self._helper = None
            self._fallback_url = self._url
            self._fallback.start(self._url)
            self._legacy_active = True
            logger.info(
                "Using compatible FFmpeg/Python motion input",
                extra={"event_type": "motion_backend_fallback", "reason": reason},
            )

    def _consume(
        self,
        command: Literal["read_motion", "discard_frame"],
        timeout_s: float,
        threshold: float | None = None,
    ) -> _MotionReply | None:
        helper = self._helper
        if helper is None or self._stopped:
            return None
        reason = "incompatible_deadline"
        if math.isfinite(timeout_s) and 0 <= timeout_s <= 120:
            reason = "motion_worker_failed"
            try:
                fields: dict[str, object] = {"wait_timeout_s": timeout_s}
                if threshold is not None:
                    fields["threshold"] = threshold
                reply = helper.request(command, timeout_s=max(0.5, timeout_s + 0.5), **fields)
                with self._lock:
                    if helper is not self._helper or self._stopped:
                        return None
                    if reply.ok:
                        frame_received = (
                            reply.observation is not None
                            if command == "read_motion"
                            else reply.frame_available
                        )
                        if frame_received:
                            self._frame_generation = self._generation
                            return reply
                        if timeout_s == 0 or self._frame_generation == self._generation:
                            return reply
                        # Native ingest may need a later keyframe. Reconnecting it at
                        # every first-frame deadline can prevent readiness indefinitely.
                        # Use the existing compatibility path before the source restarts.
                        reason = "motion_startup_timeout"
            except HelperError:
                pass
        try:
            self._start_fallback(reason, expected_helper=helper)
        except Exception:
            # Runtime startup failure is missing input, handled by the source's reconnect
            # policy. Diagnostics deliberately omit exception data from the media boundary.
            logger.warning(
                "Compatible motion input failed to start; source will reconnect",
                extra={"event_type": "motion_backend_fallback", "reason": "fallback_start_failed"},
            )
        return None

    def read_motion(self, timeout_s: float, threshold: float) -> MotionObservation | None:
        if self._legacy_active:
            return self._fallback.read_motion(timeout_s, threshold)
        generation = self._generation
        reply = self._consume("read_motion", timeout_s, threshold)
        if generation != self._generation or self._stopped:
            return None
        if self._legacy_active:
            return self._fallback.read_motion(timeout_s, threshold)
        return reply.observation if reply is not None else None

    def discard_frame(self, timeout_s: float) -> bool:
        if self._legacy_active:
            return self._fallback.discard_frame(timeout_s)
        generation = self._generation
        reply = self._consume("discard_frame", timeout_s)
        if generation != self._generation or self._stopped:
            return False
        if self._legacy_active:
            return self._fallback.discard_frame(timeout_s)
        return reply is not None and reply.frame_available

    def is_running(self) -> bool:
        if self._legacy_active:
            return self._fallback.is_running()
        helper = self._helper
        return helper is not None and helper.process.poll() is None

    def exit_code(self) -> int | None:
        if self._legacy_active:
            return self._fallback.exit_code()
        helper = self._helper
        return helper.process.poll() if helper is not None else None

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            self._generation += 1
            helper, self._helper = self._helper, None
            if helper is not None:
                helper.stop()
            self._fallback.stop()
            self._legacy_active = False


def build_motion_input(
    *,
    fallback: MotionInput,
    pixel_threshold: int,
    min_changed_pct: float,
    blur_kernel: int,
    recording_sensitivity_factor: float,
    frame_queue_size: int,
    rtsp_connect_timeout_s: float,
    rtsp_io_timeout_s: float,
    hwaccel_active: bool,
    on_frame: Callable[[], None],
    helper_path: str = "homesec-webrtc",
) -> MotionInput:
    """Reuse the existing path for settings not supported by the native CPU decoder."""
    if hwaccel_active:
        return fallback
    resolved = shutil.which(helper_path)
    if resolved is None:
        return fallback
    if (
        any(
            not math.isfinite(value) or not 0 < value <= 120
            for value in (rtsp_connect_timeout_s, rtsp_io_timeout_s)
        )
        or not 0 < frame_queue_size < 2**63
    ):
        return fallback
    try:
        settings = _MotionSettings(
            pixel_threshold=pixel_threshold,
            min_changed_pct=min_changed_pct,
            blur_kernel=blur_kernel,
            recording_sensitivity_factor=recording_sensitivity_factor,
        )
    except ValueError:
        return fallback
    return RustMotionInput(
        fallback=fallback,
        settings=settings,
        helper_path=resolved,
        frame_queue_size=frame_queue_size,
        rtsp_connect_timeout_s=rtsp_connect_timeout_s,
        rtsp_io_timeout_s=rtsp_io_timeout_s,
        on_frame=on_frame,
    )
