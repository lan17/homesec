"""Private semantic inspection contract and bounded native process ownership."""

from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from homesec.models.config import PreviewConfig, WebRTCPreviewConfig
from homesec.sources.rtsp.helper_client import (
    HelperClient,
    HelperError,
    HelperMessage,
    HelperTimeoutError,
)


def select_helper_path(preview: PreviewConfig | None) -> str:
    """Use the source's existing override, including when preview is disabled."""
    if preview is not None and isinstance(preview.config, WebRTCPreviewConfig):
        return preview.config.helper_path
    return "homesec-webrtc"


class InspectionModel(BaseModel):
    model_config = {"extra": "forbid"}


class InspectionRole(InspectionModel):
    role: str = Field(min_length=1, max_length=32)
    rtsp_url: str = Field(min_length=1, max_length=8192)
    kind: Literal["metadata", "decode", "recording"] = "decode"
    backend: str = "software"
    device: str | None = None
    audio_mode: Literal["copy", "aac", "none"] = "none"
    timestamp_mode: Literal["stream", "wallclock"] = "stream"
    output_path: str | None = None


class InspectionRequest(InspectionModel):
    request_id: str
    command: Literal[
        "stop",
        "inspect_capabilities",
        "inspect_metadata",
        "inspect_decode",
        "inspect_recording",
        "inspect_overlap",
    ]
    rtsp_url: str | None = Field(default=None, max_length=8192)
    operation_timeout_s: float | None = Field(default=None, gt=0, le=3600)
    connect_timeout_s: float | None = Field(default=None, gt=0, le=120)
    io_timeout_s: float | None = Field(default=None, gt=0, le=120)
    backend: str | None = None
    device: str | None = None
    sample_frames: int | None = Field(default=None, ge=1, le=32)
    audio_mode: Literal["copy", "aac", "none"] | None = None
    timestamp_mode: Literal["stream", "wallclock"] | None = None
    output_path: str | None = None
    duration_s: float | None = Field(default=None, gt=0, le=120)
    overlap_s: float | None = Field(default=None, gt=0, le=120)
    roles: list[InspectionRole] | None = Field(default=None, min_length=1, max_length=3)


class MetadataEvidence(InspectionModel):
    kind: Literal["metadata"]
    video_codec: str | None = None
    audio_codec: str | None = None
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    fps: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    fps_raw: str | None = None
    audio_sample_rate: int | None = None
    audio_channels: int | None = None


class DecodeEvidence(InspectionModel):
    kind: Literal["decode"]
    frames_decoded: int = Field(ge=0)
    backend: str


class StabilitySignals(InspectionModel):
    non_monotonic_dts: int = Field(default=0, ge=0)
    queue_input_backward: int = Field(default=0, ge=0)
    dts_discontinuity: int = Field(default=0, ge=0)
    sei_truncated: int = Field(default=0, ge=0)


class RecordingEvidence(InspectionModel):
    kind: Literal["recording"]
    finalized: bool
    video_packets: int = Field(ge=0)
    audio_packets: int = Field(ge=0)
    output_bytes: int = Field(ge=0)
    signals: StabilitySignals = Field(default_factory=StabilitySignals)


class RoleEvidence(InspectionModel):
    role: str
    ok: bool
    error_code: str | None = None
    frames_decoded: int = 0
    video_packets: int = 0
    audio_packets: int = 0


class OverlapEvidence(InspectionModel):
    kind: Literal["overlap"]
    status: Literal["supported", "session_limit", "failed"]
    roles: list[RoleEvidence] = Field(max_length=3)


class DeviceEvidence(InspectionModel):
    backend: str
    compiled: bool
    available: bool
    reason: str | None = None


class CapabilityEvidence(InspectionModel):
    kind: Literal["capabilities"]
    library_version: str = Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9._+ /-]+$")
    build_id: str = Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9._+-]+$")
    decoders: list[str]
    encoders: list[str]
    parsers: list[str]
    muxers: list[str]
    filters: list[str]
    hwaccels: list[str]
    device: DeviceEvidence | None = None

    def baseline_ready(self) -> bool:
        return (
            {
                "h264",
                "hevc",
                "aac",
                "aac_latm",
                "pcm_alaw",
                "pcm_mulaw",
                "opus",
                "ac3",
                "eac3",
                "alac",
                "mp2",
                "mp3",
            }.issubset(self.decoders)
            and {"aac", "libx264", "libopus"}.issubset(self.encoders)
            and {"h264", "hevc", "aac"}.issubset(self.parsers)
            and "mp4" in self.muxers
            and "aresample" in self.filters
        )


InspectionEvidence = Annotated[
    MetadataEvidence | DecodeEvidence | RecordingEvidence | OverlapEvidence | CapabilityEvidence,
    Field(discriminator="kind"),
]


class InspectionReply(HelperMessage):
    model_config = {"extra": "forbid"}
    inspection_version: int | None = None
    inspection: InspectionEvidence | None = None


class InspectionFailure(HelperError):
    """Stable secret-safe local/native failure."""

    def __init__(self, code: str, *, terminal: bool = False) -> None:
        # Do not expose arbitrary helper strings through setup responses or logs.
        safe_codes = {
            "local_capability_unavailable",
            "local_hardware_unavailable",
            "timeout",
            "authentication_failed",
            "connection_failed",
            "no_video",
            "invalid_video",
            "profile_validation_failed",
            "session_limit",
            "cancelled",
            "helper_failure",
            "cleanup_failed",
        }
        self.code = code if code in safe_codes else "helper_failure"
        self.terminal = terminal
        super().__init__(f"Native inspection: {self.code}")


class InspectionGroup:
    """One finite helper lifetime, including camera work and private trial writers.

    Cancellation is callable from another thread. Closing/reaping the helper precedes
    temporary-directory removal, so an abandoned worker cannot open another input.
    """

    CLEANUP_BUDGET_S = 0.8

    def __init__(
        self,
        *,
        helper_path: str = "homesec-webrtc",
        total_timeout_s: float,
        connect_timeout_s: float = 5.0,
        io_timeout_s: float = 5.0,
    ) -> None:
        self._helper_path = helper_path
        self._total_timeout_s = max(0.1, total_timeout_s)
        self._connect_timeout_s = max(0.001, min(120.0, connect_timeout_s))
        self._io_timeout_s = max(0.001, min(120.0, io_timeout_s))
        self._deadline: float | None = None
        self._cancelled = Event()
        self._closed = False
        self._terminal_code: str | None = None
        self._lock = Lock()
        self._helper: HelperClient[InspectionRequest, InspectionReply] | None = None
        self._temporary: TemporaryDirectory[str] | None = None

    def __enter__(self) -> InspectionGroup:
        self._deadline = time.monotonic() + self._total_timeout_s
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def cancel(self) -> None:
        self._cancelled.set()
        self.close()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            helper = self._helper
            if helper is not None:
                try:
                    helper.stop(timeout_s=self.CLEANUP_BUDGET_S)
                except HelperError as exc:
                    # Keep artifacts owned if an actual writer could still be alive.
                    raise InspectionFailure("cleanup_failed") from exc
                self._helper = None
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None

    def trial_path(self) -> Path:
        with self._lock:
            self._check_remaining()
            if self._temporary is None:
                self._temporary = TemporaryDirectory(prefix="homesec-native-inspection-")
            return Path(self._temporary.name) / "trial.mp4"

    def _check_remaining(self) -> float:
        if self._cancelled.is_set():
            raise InspectionFailure("cancelled", terminal=True)
        if self._closed:
            raise InspectionFailure(self._terminal_code or "cancelled", terminal=True)
        if self._deadline is None:
            self._deadline = time.monotonic() + self._total_timeout_s
        remaining = self._deadline - time.monotonic() - self.CLEANUP_BUDGET_S
        if remaining <= 0:
            raise InspectionFailure("timeout", terminal=True)
        return remaining

    def request(self, command: str, *, timeout_s: float, **fields: object) -> InspectionEvidence:
        try:
            with self._lock:
                remaining = self._check_remaining()
                if self._helper is None:
                    self._helper = HelperClient(
                        [self._helper_path, "--inspect"],
                        request_model=InspectionRequest,
                        reply_model=InspectionReply,
                        context="Inspection",
                    )
                    ready = self._helper.wait_ready(min(2.0, remaining))
                    if not ready.ok or ready.inspection_version != 1:
                        raise InspectionFailure("local_capability_unavailable")
                helper = self._helper
                remaining = min(timeout_s, self._check_remaining())
            # Release the owner lock while waiting, so cancellation can kill native work.
            shared: dict[str, object] = {}
            if command != "inspect_capabilities":
                shared = {
                    "operation_timeout_s": remaining,
                    "connect_timeout_s": min(self._connect_timeout_s, remaining),
                    "io_timeout_s": min(self._io_timeout_s, remaining),
                }
            reply = helper.request(command, timeout_s=remaining, **shared, **fields)
            if not reply.ok:
                raise InspectionFailure(reply.error_code or "helper_failure")
            expected_evidence = {
                "inspect_capabilities": CapabilityEvidence,
                "inspect_metadata": MetadataEvidence,
                "inspect_decode": DecodeEvidence,
                "inspect_recording": RecordingEvidence,
                "inspect_overlap": OverlapEvidence,
            }.get(command)
            if (
                reply.inspection is None
                or expected_evidence is None
                or not isinstance(reply.inspection, expected_evidence)
            ):
                raise InspectionFailure("helper_failure")
            return reply.inspection
        except (OSError, HelperError) as exc:
            fatal = (
                not isinstance(exc, InspectionFailure)
                or exc.terminal
                or exc.code
                in {
                    "helper_failure",
                    "cancelled",
                    "cleanup_failed",
                }
            )
            if fatal:
                self._terminal_code = (
                    exc.code
                    if isinstance(exc, InspectionFailure)
                    else "timeout"
                    if isinstance(exc, HelperTimeoutError)
                    else "helper_failure"
                )
                self.close()
            if isinstance(exc, InspectionFailure):
                raise
            raise InspectionFailure(self._terminal_code or "helper_failure", terminal=True) from exc
