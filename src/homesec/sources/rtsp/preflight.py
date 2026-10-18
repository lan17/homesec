from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Event, Lock
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from homesec.sources.rtsp.discovery import (
    CameraProbeResult,
    NativeStreamDiscovery,
    ProbeError,
    ProbeStreamInfo,
    build_camera_key,
)
from homesec.sources.rtsp.inspection import (
    InspectionFailure,
    InspectionGroup,
    InspectionRole,
    OverlapEvidence,
    RecordingEvidence,
)
from homesec.sources.rtsp.recording_profile import (
    MotionProfile,
    RecordingProfile,
    build_recording_profile_candidates,
)
from homesec.sources.rtsp.url_derivation import derive_probe_candidate_urls

logger = logging.getLogger(__name__)

_NON_MONOTONIC_DTS_RETRY_THRESHOLD = 5
_QUEUE_BACKWARD_RETRY_THRESHOLD = 1
_DTS_DISCONTINUITY_RETRY_THRESHOLD = 1


class ConcurrentStreamOpenResult(StrEnum):
    SUPPORTED = "supported"
    SESSION_LIMIT = "session_limit"
    FAILED = "failed"


class StreamDiscovery(Protocol):
    def probe(
        self,
        *,
        camera_key: str,
        candidate_urls: list[str],
    ) -> CameraProbeResult | ProbeError: ...


class SelectionError(Exception):
    def __init__(self, camera_key: str, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.camera_key = camera_key
        self.message = message
        self.cause = cause
        self.__cause__ = cause


class NegotiationError(Exception):
    def __init__(self, camera_key: str, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.camera_key = camera_key
        self.message = message
        self.cause = cause
        self.__cause__ = cause


class PreflightError(Exception):
    def __init__(
        self,
        *,
        camera_key: str,
        stage: str,
        message: str,
        diagnostics: CameraPreflightDiagnostics | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.camera_key = camera_key
        self.stage = stage
        self.message = message
        self.diagnostics = diagnostics
        self.cause = cause
        self.__cause__ = cause


class CameraPreflightDiagnostics(BaseModel):
    """Internal startup decisions; private URLs must not be logged or persisted."""

    model_config = {"extra": "forbid"}

    attempted_urls: list[str]
    probes: list[ProbeStreamInfo]
    selected_motion_url: str | None = None
    selected_recording_url: str | None = None
    selected_preview_url: str | None = None
    selected_preview_probe_url: str | None = None
    negotiation_attempts: list[str] = Field(default_factory=list)
    selected_recording_profile: str | None = None
    session_mode: Literal["dual_stream", "single_stream"] | None = None
    concurrent_preview_supported: bool | None = None
    concurrent_preview_downgrade_reason: str | None = None
    notes: list[str] = Field(default_factory=list)


class CameraPreflightOutcome(BaseModel):
    """Locked startup preflight result for one RTSP camera."""

    model_config = {"extra": "forbid"}

    camera_key: str
    motion_profile: MotionProfile
    recording_profile: RecordingProfile
    concurrent_preview_supported: bool | None = None
    concurrent_preview_downgrade_reason: str | None = None
    diagnostics: CameraPreflightDiagnostics


class RecordingValidationSignals(BaseModel):
    """Bounded timing evidence from native recording validation."""

    model_config = {"extra": "forbid"}

    non_monotonic_dts: int = 0
    queue_input_backward: int = 0
    dts_discontinuity: int = 0
    sei_truncated: int = 0

    def instability_score(self) -> int:
        # Penalize explicit backward-queue and discontinuity warnings first.
        return (
            (self.queue_input_backward * 1_000)
            + (self.dts_discontinuity * 100)
            + self.non_monotonic_dts
        )

    def needs_wallclock_retry(self) -> bool:
        return (
            self.queue_input_backward >= _QUEUE_BACKWARD_RETRY_THRESHOLD
            or self.non_monotonic_dts >= _NON_MONOTONIC_DTS_RETRY_THRESHOLD
            or self.dts_discontinuity >= _DTS_DISCONTINUITY_RETRY_THRESHOLD
        )


class RecordingValidationResult(BaseModel):
    """Result for one recording-profile validation attempt."""

    model_config = {"extra": "forbid"}

    ok: bool
    error: str | None = None
    signals: RecordingValidationSignals = Field(default_factory=RecordingValidationSignals)


@dataclass(frozen=True, slots=True)
class _SessionOpenSpec:
    role: str
    input_url: str
    kind: Literal["decode", "metadata"] = "decode"
    include_audio: bool = False


class RTSPStartupPreflight:
    """Startup-only RTSP preflight: discovery, selection, and negotiation."""

    def __init__(
        self,
        *,
        output_dir: Path,
        rtsp_connect_timeout_s: float,
        rtsp_io_timeout_s: float,
        discovery: StreamDiscovery | None = None,
        session_overlap_s: float = 1.0,
        command_timeout_s: float = 8.0,
        inspection: InspectionGroup | None = None,
        helper_path: str = "homesec-webrtc",
        total_timeout_s: float | None = None,
        stop_event: Event | None = None,
    ) -> None:
        self._output_dir = output_dir
        self._rtsp_connect_timeout_s = rtsp_connect_timeout_s
        self._rtsp_io_timeout_s = rtsp_io_timeout_s
        self._session_overlap_s = max(0.1, session_overlap_s)
        self._command_timeout_s = max(1.0, command_timeout_s)
        self._inspection = inspection
        self._helper_path = helper_path
        self._total_timeout_s = total_timeout_s
        self._discovery = discovery
        self._cancelled = stop_event if stop_event is not None else Event()
        self._owner_lock = Lock()

    def cancel(self) -> None:
        self._cancelled.set()
        with self._owner_lock:
            inspection = self._inspection
        if inspection is not None:
            inspection.cancel()

    def run(
        self,
        *,
        camera_name: str,
        primary_rtsp_url: str,
        detect_rtsp_url: str,
        preview_rtsp_url: str | None = None,
        preview_probe_rtsp_url: str | None = None,
        preview_audio_enabled: bool = False,
    ) -> CameraPreflightOutcome | PreflightError:
        # One cap applies to each operation; the finite plan determines group lifetime.
        candidate_count = len(_candidate_urls(primary_rtsp_url, detect_rtsp_url))
        overlap_count = (
            2 + int(preview_rtsp_url is not None) + int(preview_probe_rtsp_url is not None)
        )
        overlap_cap = self._command_timeout_s + (3 * max(2.0, self._session_overlap_s + 0.5))
        total = self._total_timeout_s or (
            (candidate_count + 6) * self._command_timeout_s + overlap_count * overlap_cap + 3.0
        )
        with self._owner_lock:
            if self._cancelled.is_set():
                return PreflightError(
                    camera_key=build_camera_key(camera_name, primary_rtsp_url),
                    stage="inspection",
                    message="Native inspection: cancelled",
                )
            own_inspection = self._inspection is None
            if own_inspection:
                self._inspection = InspectionGroup(
                    helper_path=self._helper_path,
                    total_timeout_s=total,
                    connect_timeout_s=self._rtsp_connect_timeout_s,
                    io_timeout_s=self._rtsp_io_timeout_s,
                )
        try:
            return self._run(
                camera_name=camera_name,
                primary_rtsp_url=primary_rtsp_url,
                detect_rtsp_url=detect_rtsp_url,
                preview_rtsp_url=preview_rtsp_url,
                preview_probe_rtsp_url=preview_probe_rtsp_url,
                preview_audio_enabled=preview_audio_enabled,
            )
        except InspectionFailure as exc:
            return PreflightError(
                camera_key=build_camera_key(camera_name, primary_rtsp_url),
                stage="inspection",
                message=str(exc),
                cause=exc,
            )
        finally:
            if own_inspection and self._inspection is not None:
                self._inspection.close()
                with self._owner_lock:
                    self._inspection = None

    def _run(
        self,
        *,
        camera_name: str,
        primary_rtsp_url: str,
        detect_rtsp_url: str,
        preview_rtsp_url: str | None = None,
        preview_probe_rtsp_url: str | None = None,
        preview_audio_enabled: bool = False,
    ) -> CameraPreflightOutcome | PreflightError:
        """Run startup preflight and return locked profiles or a typed error."""

        camera_key = build_camera_key(camera_name, primary_rtsp_url)
        candidate_urls = _candidate_urls(primary_rtsp_url, detect_rtsp_url)

        logger.info("RTSP preflight started: camera=%s", camera_name)

        discovery = self._discovery or NativeStreamDiscovery(
            rtsp_connect_timeout_s=self._rtsp_connect_timeout_s,
            rtsp_io_timeout_s=self._rtsp_io_timeout_s,
            command_timeout_s=self._command_timeout_s,
            inspection=self._inspection,
            helper_path=self._helper_path,
        )
        probe_result = discovery.probe(camera_key=camera_key, candidate_urls=candidate_urls)
        match probe_result:
            case ProbeError() as probe_error:
                return PreflightError(
                    camera_key=camera_key,
                    stage="inspection"
                    if isinstance(probe_error.cause, InspectionFailure)
                    else "probe",
                    message=probe_error.message,
                    cause=probe_error,
                )
            case CameraProbeResult() as probe_data:
                pass
            case _:
                return PreflightError(
                    camera_key=camera_key,
                    stage="probe",
                    message=(f"Unexpected discovery result type: {type(probe_result).__name__}"),
                )

        diagnostics = CameraPreflightDiagnostics(
            attempted_urls=probe_data.attempted_urls,
            probes=probe_data.streams,
        )

        local_failure = next(
            (
                stream.error
                for stream in probe_data.streams
                if stream.error
                in {
                    "Native inspection: local_capability_unavailable",
                    "Native inspection: helper_failure",
                    "Native inspection: cleanup_failed",
                    "Native inspection: cancelled",
                }
            ),
            None,
        )
        if local_failure is not None:
            return PreflightError(
                camera_key=camera_key,
                stage="inspection",
                message=local_failure,
                diagnostics=diagnostics,
            )
        viable_streams = [stream for stream in probe_data.streams if stream.probe_ok]
        if not viable_streams:
            return PreflightError(
                camera_key=camera_key,
                stage="selection",
                message="No viable RTSP streams found during preflight",
                diagnostics=diagnostics,
            )

        try:
            motion_stream = select_motion_stream(camera_key=camera_key, streams=viable_streams)
            recording_stream = select_recording_stream(
                camera_key=camera_key, streams=viable_streams
            )
        except SelectionError as selection_error:
            return PreflightError(
                camera_key=camera_key,
                stage="selection",
                message=selection_error.message,
                diagnostics=diagnostics,
                cause=selection_error,
            )

        diagnostics.selected_motion_url = motion_stream.url
        diagnostics.selected_recording_url = recording_stream.url
        diagnostics.selected_preview_url = preview_rtsp_url
        diagnostics.selected_preview_probe_url = preview_probe_rtsp_url

        negotiation = self._negotiate_recording_profile(
            camera_key=camera_key,
            stream=recording_stream,
            diagnostics=diagnostics,
        )
        match negotiation:
            case NegotiationError() as negotiation_error:
                return PreflightError(
                    camera_key=camera_key,
                    stage="negotiation",
                    message=negotiation_error.message,
                    diagnostics=diagnostics,
                    cause=negotiation_error,
                )
            case RecordingProfile() as recording_profile:
                pass
            case _:
                return PreflightError(
                    camera_key=camera_key,
                    stage="negotiation",
                    message=(f"Unexpected negotiation result type: {type(negotiation).__name__}"),
                    diagnostics=diagnostics,
                )

        motion_profile = MotionProfile(input_url=motion_stream.url, ffmpeg_input_args=[])

        if self._validate_session_limits(motion_profile.input_url, recording_profile.input_url):
            diagnostics.session_mode = "dual_stream"
            self._classify_concurrent_preview(
                motion_profile=motion_profile,
                recording_profile=recording_profile,
                preview_rtsp_url=preview_rtsp_url,
                preview_probe_rtsp_url=preview_probe_rtsp_url,
                preview_audio_enabled=preview_audio_enabled,
                diagnostics=diagnostics,
            )
            return CameraPreflightOutcome(
                camera_key=camera_key,
                motion_profile=motion_profile,
                recording_profile=recording_profile,
                concurrent_preview_supported=diagnostics.concurrent_preview_supported,
                concurrent_preview_downgrade_reason=(
                    diagnostics.concurrent_preview_downgrade_reason
                ),
                diagnostics=diagnostics,
            )

        # Session-limited fallback: use one stream URL for both roles.
        motion_profile = MotionProfile(input_url=recording_profile.input_url, ffmpeg_input_args=[])
        diagnostics.selected_motion_url = recording_profile.input_url
        diagnostics.notes.append("Dual-stream open failed; retrying in single-stream mode")

        if not self._validate_session_limits(motion_profile.input_url, recording_profile.input_url):
            return PreflightError(
                camera_key=camera_key,
                stage="session_limit",
                message=(
                    "RTSP concurrent session validation failed for both dual-stream and "
                    "single-stream modes"
                ),
                diagnostics=diagnostics,
            )

        diagnostics.session_mode = "single_stream"
        self._classify_concurrent_preview(
            motion_profile=motion_profile,
            recording_profile=recording_profile,
            preview_rtsp_url=preview_rtsp_url,
            preview_probe_rtsp_url=preview_probe_rtsp_url,
            preview_audio_enabled=preview_audio_enabled,
            diagnostics=diagnostics,
        )
        return CameraPreflightOutcome(
            camera_key=camera_key,
            motion_profile=motion_profile,
            recording_profile=recording_profile,
            concurrent_preview_supported=diagnostics.concurrent_preview_supported,
            concurrent_preview_downgrade_reason=diagnostics.concurrent_preview_downgrade_reason,
            diagnostics=diagnostics,
        )

    def _classify_concurrent_preview(
        self,
        *,
        motion_profile: MotionProfile,
        recording_profile: RecordingProfile,
        preview_rtsp_url: str | None,
        preview_probe_rtsp_url: str | None,
        preview_audio_enabled: bool,
        diagnostics: CameraPreflightDiagnostics,
    ) -> None:
        if preview_rtsp_url is None:
            return

        if preview_probe_rtsp_url is not None:
            probe_validation_result = self._validate_concurrent_preview_probe_session_limits(
                motion_profile.input_url,
                recording_profile.input_url,
                preview_probe_rtsp_url,
            )
            if probe_validation_result is ConcurrentStreamOpenResult.SESSION_LIMIT:
                self._mark_concurrent_preview_unsupported(diagnostics)
                return
            if probe_validation_result is not ConcurrentStreamOpenResult.SUPPORTED:
                diagnostics.notes.append(
                    "Concurrent preview startup probe could not be classified during startup; "
                    "continuing with preview stream validation"
                )
            else:
                diagnostics.notes.append(
                    "Concurrent preview startup probe validated with motion and recording streams"
                )

        validation_result = self._validate_concurrent_preview_session_limits(
            motion_profile.input_url,
            recording_profile.input_url,
            preview_rtsp_url,
            preview_audio_enabled=preview_audio_enabled,
        )
        if validation_result is ConcurrentStreamOpenResult.SUPPORTED:
            diagnostics.concurrent_preview_supported = True
            diagnostics.notes.append(
                "Concurrent preview while recording validated with motion, recording, and preview streams"
            )
            return

        if validation_result is not ConcurrentStreamOpenResult.SESSION_LIMIT:
            diagnostics.notes.append(
                "Concurrent preview while recording could not be classified during startup; "
                "runtime will continue best-effort behavior"
            )
            return

        self._mark_concurrent_preview_unsupported(diagnostics)

    @staticmethod
    def _mark_concurrent_preview_unsupported(
        diagnostics: CameraPreflightDiagnostics,
    ) -> None:
        diagnostics.concurrent_preview_supported = False
        diagnostics.concurrent_preview_downgrade_reason = (
            "concurrent_preview_unsupported_by_startup_preflight"
        )
        diagnostics.notes.append(
            "Concurrent preview while recording failed startup validation; "
            "preview will use recording-first behavior for this process"
        )

    def _negotiate_recording_profile(
        self,
        *,
        camera_key: str,
        stream: ProbeStreamInfo,
        diagnostics: CameraPreflightDiagnostics,
    ) -> RecordingProfile | NegotiationError:
        candidates = build_recording_profile_candidates(
            input_url=stream.url,
            audio_codec=stream.audio_codec,
        )

        last_error: str | None = None
        for candidate in candidates:
            diagnostics.negotiation_attempts.append(candidate.profile_id())
            validation = self._validate_recording_profile(candidate)
            if validation.ok:
                selected_profile = self._maybe_enable_wallclock_timestamps(
                    candidate=candidate,
                    validation=validation,
                    diagnostics=diagnostics,
                )
                diagnostics.selected_recording_profile = selected_profile.profile_id()
                logger.info(
                    "Recording profile selected: profile=%s",
                    selected_profile.profile_id(),
                )
                return selected_profile

            if validation.error:
                last_error = validation.error

        error_text = last_error or "No candidate profile succeeded"
        return NegotiationError(
            camera_key,
            f"Recording profile negotiation failed: {error_text}",
        )

    def _maybe_enable_wallclock_timestamps(
        self,
        *,
        candidate: RecordingProfile,
        validation: RecordingValidationResult,
        diagnostics: CameraPreflightDiagnostics,
    ) -> RecordingProfile:
        if candidate.uses_wallclock_timestamps():
            return candidate
        if not validation.signals.needs_wallclock_retry():
            return candidate

        wallclock_candidate = candidate.model_copy(
            update={"ffmpeg_input_args": ["-use_wallclock_as_timestamps", "1"]}
        )
        diagnostics.negotiation_attempts.append(wallclock_candidate.profile_id())
        wallclock_validation = self._validate_recording_profile(wallclock_candidate)

        if not wallclock_validation.ok:
            diagnostics.notes.append(
                "Wallclock retry failed; keeping baseline recording timestamp mode"
            )
            return candidate

        baseline_score = validation.signals.instability_score()
        wallclock_score = wallclock_validation.signals.instability_score()
        if wallclock_score < baseline_score:
            diagnostics.notes.append(
                "Wallclock timestamps enabled after startup validation reduced DTS instability"
            )
            return wallclock_candidate

        return candidate

    def _validate_recording_profile(self, profile: RecordingProfile) -> RecordingValidationResult:
        inspection = self._inspection
        if inspection is None:
            raise InspectionFailure("helper_failure")
        target = inspection.trial_path()
        try:
            evidence = inspection.request(
                "inspect_recording",
                timeout_s=self._command_timeout_s,
                rtsp_url=profile.input_url,
                audio_mode=profile.audio_mode,
                timestamp_mode="wallclock" if profile.uses_wallclock_timestamps() else "stream",
                output_path=str(target),
                duration_s=2.0,
            )
            if not isinstance(evidence, RecordingEvidence):
                raise InspectionFailure("helper_failure")
            signals = RecordingValidationSignals.model_validate(evidence.signals.model_dump())
            valid = (
                evidence.finalized
                and evidence.video_packets > 0
                and evidence.output_bytes > 0
                and target.is_file()
                and target.stat().st_size > 0
                and (profile.audio_mode == "none" or evidence.audio_packets > 0)
            )
            return RecordingValidationResult(
                ok=valid,
                signals=signals,
                error=None if valid else "Native inspection: profile_validation_failed",
            )
        except InspectionFailure as exc:
            # Missing local components must not silently negotiate audio away.
            if exc.terminal or exc.code in {
                "local_capability_unavailable",
                "helper_failure",
                "cancelled",
            }:
                raise
            return RecordingValidationResult(ok=False, error=str(exc))
        finally:
            # request failures terminate the owner before this writer target is removed.
            target.unlink(missing_ok=True)

    def _validate_session_limits(self, motion_url: str, recording_url: str) -> bool:
        """Validate that motion and recording pipelines can open concurrently."""
        return (
            self._validate_concurrent_stream_opens(
                [
                    _SessionOpenSpec(role="motion", input_url=motion_url),
                    _SessionOpenSpec(role="recording", input_url=recording_url),
                ]
            )
            is ConcurrentStreamOpenResult.SUPPORTED
        )

    def _validate_concurrent_preview_session_limits(
        self,
        motion_url: str,
        recording_url: str,
        preview_url: str,
        *,
        preview_audio_enabled: bool = False,
    ) -> ConcurrentStreamOpenResult:
        """Validate the current runtime topology for motion, recording, and preview."""
        return self._validate_concurrent_stream_opens(
            [
                _SessionOpenSpec(role="motion", input_url=motion_url),
                _SessionOpenSpec(role="recording", input_url=recording_url),
                _SessionOpenSpec(
                    role="preview",
                    input_url=preview_url,
                    include_audio=preview_audio_enabled,
                ),
            ]
        )

    def _validate_concurrent_preview_probe_session_limits(
        self,
        motion_url: str,
        recording_url: str,
        preview_probe_url: str,
    ) -> ConcurrentStreamOpenResult:
        """Validate the preview startup probe alongside motion and recording."""
        return self._validate_concurrent_stream_opens(
            [
                _SessionOpenSpec(role="motion", input_url=motion_url),
                _SessionOpenSpec(role="recording", input_url=recording_url),
                _SessionOpenSpec(
                    role="preview_probe",
                    input_url=preview_probe_url,
                    kind="metadata",
                ),
            ]
        )

    def _validate_concurrent_stream_opens(
        self, specs: list[_SessionOpenSpec]
    ) -> ConcurrentStreamOpenResult:
        inspection = self._inspection
        if inspection is None:
            raise InspectionFailure("helper_failure")
        roles = [
            InspectionRole(
                role=spec.role,
                rtsp_url=spec.input_url,
                kind=spec.kind,
                audio_mode="copy" if spec.include_audio else "none",
            )
            for spec in specs
        ]
        try:
            evidence = inspection.request(
                "inspect_overlap",
                timeout_s=self._command_timeout_s
                + len(roles) * max(2.0, self._session_overlap_s + 0.5),
                roles=roles,
                overlap_s=self._session_overlap_s,
                duration_s=max(2.0, self._session_overlap_s + 0.5),
            )
            if not isinstance(evidence, OverlapEvidence):
                raise InspectionFailure("helper_failure")
            if any(role.error_code == "local_capability_unavailable" for role in evidence.roles):
                raise InspectionFailure("local_capability_unavailable")
            if evidence.status == "session_limit":
                return ConcurrentStreamOpenResult.SESSION_LIMIT
            # A status string alone cannot prove that every independent role completed.
            if (
                evidence.status == "supported"
                and {role.role for role in evidence.roles} == {role.role for role in roles}
                and len(evidence.roles) == len(roles)
                and all(
                    role.ok and (role.frames_decoded > 0 or role.video_packets > 0)
                    for role in evidence.roles
                )
            ):
                return ConcurrentStreamOpenResult.SUPPORTED
            return ConcurrentStreamOpenResult.FAILED
        except InspectionFailure as exc:
            if exc.code == "session_limit":
                return ConcurrentStreamOpenResult.SESSION_LIMIT
            if exc.terminal or exc.code in {
                "local_capability_unavailable",
                "helper_failure",
                "cancelled",
            }:
                raise
            return ConcurrentStreamOpenResult.FAILED


def select_motion_stream(*, camera_key: str, streams: list[ProbeStreamInfo]) -> ProbeStreamInfo:
    if not streams:
        raise SelectionError(camera_key, "No streams available for motion selection")

    def sort_key(stream: ProbeStreamInfo) -> tuple[float, float, float, str]:
        width = float(stream.width) if stream.width is not None else float("inf")
        height = float(stream.height) if stream.height is not None else float("inf")
        fps = stream.fps if stream.fps is not None else float("inf")
        return (width, height, fps, stream.url)

    return sorted(streams, key=sort_key)[0]


def select_recording_stream(*, camera_key: str, streams: list[ProbeStreamInfo]) -> ProbeStreamInfo:
    if not streams:
        raise SelectionError(camera_key, "No streams available for recording selection")

    def sort_key(stream: ProbeStreamInfo) -> tuple[float, float, float, str]:
        width = float(stream.width) if stream.width is not None else 0.0
        height = float(stream.height) if stream.height is not None else 0.0
        fps = stream.fps if stream.fps is not None else 0.0
        return (-width, -height, -fps, stream.url)

    return sorted(streams, key=sort_key)[0]


def _candidate_urls(primary_rtsp_url: str, detect_rtsp_url: str) -> list[str]:
    candidates: list[str] = [primary_rtsp_url]
    if detect_rtsp_url != primary_rtsp_url:
        candidates.append(detect_rtsp_url)

    for base_url in (primary_rtsp_url, detect_rtsp_url):
        for derived_url in derive_probe_candidate_urls(base_url):
            if derived_url in candidates:
                continue
            candidates.append(derived_url)

    return candidates
