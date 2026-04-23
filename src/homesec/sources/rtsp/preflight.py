from __future__ import annotations

import logging
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from homesec.sources.rtsp.capabilities import (
    RTSPTimeoutCapabilities,
    get_global_rtsp_timeout_capabilities,
)
from homesec.sources.rtsp.discovery import (
    CameraProbeResult,
    FfprobeStreamDiscovery,
    ProbeError,
    ProbeStreamInfo,
    build_camera_key,
)
from homesec.sources.rtsp.recording_profile import (
    MotionProfile,
    RecordingProfile,
    build_recording_profile_candidates,
)
from homesec.sources.rtsp.url_derivation import derive_probe_candidate_urls
from homesec.sources.rtsp.utils import (
    _build_timeout_attempts,
    _format_cmd,
    _is_timeout_option_error,
    _redact_rtsp_url,
    _signal_process_group,
)

logger = logging.getLogger(__name__)

_NON_MONOTONIC_DTS_RETRY_THRESHOLD = 5
_QUEUE_BACKWARD_RETRY_THRESHOLD = 1
_DTS_DISCONTINUITY_RETRY_THRESHOLD = 1
_SESSION_LIMIT_HINTS: tuple[str, ...] = (
    "too many clients",
    "too many connections",
    "too many sessions",
    "max number of clients",
    "maximum number of clients",
    "maximum number of connections",
    "session limit",
    "max sessions",
    "maximum sessions",
)


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
    """Secret-safe summary of startup preflight decisions and attempts."""

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
    """Stability signals extracted from ffmpeg/ffprobe stderr text."""

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
    tool: Literal["ffmpeg", "ffprobe"] = "ffmpeg"


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
        timeout_capabilities: RTSPTimeoutCapabilities | None = None,
    ) -> None:
        self._output_dir = output_dir
        self._rtsp_connect_timeout_s = rtsp_connect_timeout_s
        self._rtsp_io_timeout_s = rtsp_io_timeout_s
        self._session_overlap_s = max(0.1, session_overlap_s)
        self._command_timeout_s = max(1.0, command_timeout_s)
        self._timeout_capabilities = timeout_capabilities or get_global_rtsp_timeout_capabilities()
        self._discovery = discovery or FfprobeStreamDiscovery(
            rtsp_connect_timeout_s=rtsp_connect_timeout_s,
            rtsp_io_timeout_s=rtsp_io_timeout_s,
            command_timeout_s=command_timeout_s,
            timeout_capabilities=self._timeout_capabilities,
        )

    def run(
        self,
        *,
        camera_name: str,
        primary_rtsp_url: str,
        detect_rtsp_url: str,
        preview_rtsp_url: str | None = None,
        preview_probe_rtsp_url: str | None = None,
    ) -> CameraPreflightOutcome | PreflightError:
        """Run startup preflight and return locked profiles or a typed error."""

        camera_key = build_camera_key(camera_name, primary_rtsp_url)
        candidate_urls = _candidate_urls(primary_rtsp_url, detect_rtsp_url)

        logger.info("RTSP preflight started: camera=%s key=%s", camera_name, camera_key)

        probe_result = self._discovery.probe(camera_key=camera_key, candidate_urls=candidate_urls)
        match probe_result:
            case ProbeError() as probe_error:
                return PreflightError(
                    camera_key=camera_key,
                    stage="probe",
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
                    "runtime will continue best-effort behavior"
                )
                return
            diagnostics.notes.append(
                "Concurrent preview startup probe validated with motion and recording streams"
            )

        validation_result = self._validate_concurrent_preview_session_limits(
            motion_profile.input_url,
            recording_profile.input_url,
            preview_rtsp_url,
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
                    "Recording profile selected: camera=%s profile=%s",
                    camera_key,
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
        """Run short ffmpeg check to validate locked recording profile."""

        self._output_dir.mkdir(parents=True, exist_ok=True)

        with NamedTemporaryFile(
            mode="wb",
            prefix="preflight_",
            suffix=f".{profile.output_extension}",
            dir=self._output_dir,
            delete=False,
        ) as tmp_file:
            temp_path = Path(tmp_file.name)

        timeout_args = self._timeout_args()

        attempts = _build_timeout_attempts(timeout_args)

        try:
            for label, extra_timeout_args in attempts:
                cmd = self._build_recording_check_cmd(
                    input_url=profile.input_url,
                    input_args=profile.ffmpeg_input_args,
                    output_path=temp_path,
                    output_args=profile.ffmpeg_output_args,
                    timeout_args=extra_timeout_args,
                )

                redacted_cmd = _format_recording_cmd(cmd)
                logger.debug("Recording preflight command (%s): %s", label, redacted_cmd)

                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=self._command_timeout_s,
                        check=False,
                    )
                except Exception as exc:
                    return RecordingValidationResult(
                        ok=False, error=f"ffmpeg execution failed: {type(exc).__name__}"
                    )

                stderr_text = _sanitize_stderr(result.stderr, profile.input_url)
                signals = _collect_recording_stability_signals(stderr_text)
                if result.returncode == 0:
                    if label == "timeouts":
                        self._timeout_capabilities.note_ffmpeg_timeout_success()
                    try:
                        exists = temp_path.exists()
                    except Exception as exc:
                        logger.warning(
                            "Recording preflight failed to verify output existence: %s",
                            exc,
                            exc_info=True,
                        )
                        return RecordingValidationResult(
                            ok=False,
                            error=f"failed to verify output file existence: {type(exc).__name__}",
                            signals=signals,
                        )
                    if not exists:
                        return RecordingValidationResult(
                            ok=False,
                            error="ffmpeg returned success but output clip was not created",
                            signals=signals,
                        )
                    try:
                        size = temp_path.stat().st_size
                    except Exception as exc:
                        logger.warning(
                            "Recording preflight failed to stat output clip: %s",
                            exc,
                            exc_info=True,
                        )
                        return RecordingValidationResult(
                            ok=False,
                            error=f"failed to stat output clip: {type(exc).__name__}",
                            signals=signals,
                        )
                    if size > 0:
                        return RecordingValidationResult(ok=True, signals=signals)
                    return RecordingValidationResult(
                        ok=False,
                        error="ffmpeg returned success but output clip is empty",
                        signals=signals,
                    )

                if label == "timeouts" and _is_timeout_option_error(stderr_text):
                    changed = self._timeout_capabilities.mark_ffmpeg_timeout_unsupported()
                    if changed:
                        logger.warning(
                            "ffmpeg timeout options unsupported; disabling RTSP timeout options for this process"
                        )
                    logger.debug("Recording preflight retrying without timeout options")
                    continue

                return RecordingValidationResult(
                    ok=False,
                    error=stderr_text[:280] or f"ffmpeg failed ({result.returncode})",
                    signals=signals,
                )

            return RecordingValidationResult(
                ok=False,
                error="ffmpeg failed after timeout-option fallback",
            )
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                logger.debug("Failed to remove preflight temp file: %s", temp_path)

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
    ) -> ConcurrentStreamOpenResult:
        """Validate the current runtime topology for motion, recording, and preview."""
        return self._validate_concurrent_stream_opens(
            [
                _SessionOpenSpec(role="motion", input_url=motion_url),
                _SessionOpenSpec(role="recording", input_url=recording_url),
                _SessionOpenSpec(role="preview", input_url=preview_url),
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
                    tool="ffprobe",
                ),
            ]
        )

    def _validate_concurrent_stream_opens(
        self, specs: list[_SessionOpenSpec]
    ) -> ConcurrentStreamOpenResult:
        """Validate that the requested RTSP consumers can open concurrently."""
        attempts = self._stream_open_attempts(specs)
        session_check_duration_s = max(2.0, self._session_overlap_s + 0.5)
        completion_timeout_s = session_check_duration_s + 1.5

        for label, use_timeout_args in attempts:
            procs: dict[str, subprocess.Popen[str] | None] = {spec.role: None for spec in specs}
            specs_by_role = {spec.role: spec for spec in specs}

            try:
                for spec in specs:
                    timeout_args = (
                        self._timeout_args_for_tool(spec.tool) if use_timeout_args else []
                    )
                    cmd = self._build_session_open_cmd(
                        spec=spec,
                        timeout_args=timeout_args,
                        duration_s=session_check_duration_s,
                    )
                    procs[spec.role] = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                        start_new_session=True,
                    )
            except Exception as exc:
                logger.warning(
                    "Session-limit validation failed to launch ffmpeg (%s): %s",
                    label,
                    exc,
                    exc_info=True,
                )
                for proc in procs.values():
                    self._terminate_process(proc)
                continue

            time.sleep(self._session_overlap_s)

            initial_exits = {
                role: proc.poll() if proc is not None else None for role, proc in procs.items()
            }
            if all(exit_code is None for exit_code in initial_exits.values()):
                exits_and_errors = {
                    role: self._wait_for_process_exit(
                        proc,
                        timeout_s=completion_timeout_s,
                    )
                    for role, proc in procs.items()
                }
            else:
                exits_and_errors = {
                    role: self._terminate_process(proc) for role, proc in procs.items()
                }

            if all(exit_code == 0 for exit_code, _err in exits_and_errors.values()):
                if label == "timeouts":
                    self._note_timeout_success(specs)
                return ConcurrentStreamOpenResult.SUPPORTED

            timeout_option_error = label == "timeouts" and any(
                _is_timeout_option_error(err) for _exit_code, err in exits_and_errors.values()
            )
            if any(_is_session_limit_error(err) for _exit_code, err in exits_and_errors.values()):
                return ConcurrentStreamOpenResult.SESSION_LIMIT

            if timeout_option_error:
                self._mark_timeout_unsupported(specs_by_role, exits_and_errors)
                logger.debug(
                    "Session-limit validation retrying without timeout options; errors=%s",
                    {role: err for role, (_exit_code, err) in exits_and_errors.items() if err},
                )
                continue

            exits = {role: exit_code for role, (exit_code, _err) in exits_and_errors.items()}
            errors = {role: err for role, (_exit_code, err) in exits_and_errors.items() if err}
            logger.warning(
                "Session-limit validation failed (%s): exits=%s errors=%s",
                label,
                exits,
                errors,
            )
            if label == "timeouts":
                return ConcurrentStreamOpenResult.FAILED

        return ConcurrentStreamOpenResult.FAILED

    def _build_recording_check_cmd(
        self,
        *,
        input_url: str,
        input_args: list[str],
        output_path: Path,
        output_args: list[str],
        timeout_args: list[str],
    ) -> list[str]:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "warning",
            "-rtsp_transport",
            "tcp",
        ]
        cmd.extend(timeout_args)
        cmd.extend(input_args)
        cmd.extend(
            [
                "-i",
                input_url,
                "-t",
                "2",
            ]
        )
        cmd.extend(output_args)
        cmd.extend(["-y", str(output_path)])
        return cmd

    def _stream_open_attempts(
        self,
        specs: list[_SessionOpenSpec],
    ) -> tuple[tuple[Literal["timeouts", "no_timeouts"], bool], ...]:
        has_timeout_args = any(self._timeout_args_for_tool(spec.tool) for spec in specs)
        if has_timeout_args:
            return (("timeouts", True), ("no_timeouts", False))
        return (("no_timeouts", False),)

    def _build_session_open_cmd(
        self,
        *,
        spec: _SessionOpenSpec,
        timeout_args: list[str],
        duration_s: float,
    ) -> list[str]:
        match spec.tool:
            case "ffmpeg":
                return self._build_stream_open_cmd(
                    input_url=spec.input_url,
                    timeout_args=timeout_args,
                    duration_s=duration_s,
                )
            case "ffprobe":
                return self._build_probe_open_cmd(
                    input_url=spec.input_url,
                    timeout_args=timeout_args,
                    duration_s=duration_s,
                )

    def _build_stream_open_cmd(
        self,
        *,
        input_url: str,
        timeout_args: list[str],
        duration_s: float,
    ) -> list[str]:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
        ]
        cmd.extend(timeout_args)
        cmd.extend(
            [
                "-i",
                input_url,
                "-t",
                f"{max(0.5, duration_s):.1f}",
                "-an",
                "-f",
                "null",
                "-",
            ]
        )
        return cmd

    def _build_probe_open_cmd(
        self,
        *,
        input_url: str,
        timeout_args: list[str],
        duration_s: float,
    ) -> list[str]:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-read_intervals",
            f"%+{max(0.5, duration_s):.1f}",
            "-count_packets",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_packets",
            "-of",
            "json",
            "-rtsp_transport",
            "tcp",
        ]
        cmd.extend(timeout_args)
        cmd.append(input_url)
        return cmd

    def _timeout_args(self) -> list[str]:
        return self._timeout_capabilities.build_ffmpeg_timeout_args(
            connect_timeout_s=self._rtsp_connect_timeout_s,
            io_timeout_s=self._rtsp_io_timeout_s,
        )

    def _ffprobe_timeout_args(self) -> list[str]:
        return self._timeout_capabilities.build_ffprobe_timeout_args(
            connect_timeout_s=self._rtsp_connect_timeout_s,
            io_timeout_s=self._rtsp_io_timeout_s,
        )

    def _timeout_args_for_tool(self, tool: Literal["ffmpeg", "ffprobe"]) -> list[str]:
        match tool:
            case "ffmpeg":
                return self._timeout_args()
            case "ffprobe":
                return self._ffprobe_timeout_args()

    def _note_timeout_success(self, specs: list[_SessionOpenSpec]) -> None:
        if any(spec.tool == "ffmpeg" and self._timeout_args() for spec in specs):
            self._timeout_capabilities.note_ffmpeg_timeout_success()
        if any(spec.tool == "ffprobe" and self._ffprobe_timeout_args() for spec in specs):
            self._timeout_capabilities.note_ffprobe_timeout_success()

    def _mark_timeout_unsupported(
        self,
        specs_by_role: dict[str, _SessionOpenSpec],
        exits_and_errors: dict[str, tuple[int | None, str]],
    ) -> None:
        ffmpeg_changed = False
        ffprobe_changed = False
        for role, (_exit_code, err) in exits_and_errors.items():
            if not _is_timeout_option_error(err):
                continue
            spec = specs_by_role[role]
            match spec.tool:
                case "ffmpeg":
                    ffmpeg_changed = (
                        self._timeout_capabilities.mark_ffmpeg_timeout_unsupported()
                        or ffmpeg_changed
                    )
                case "ffprobe":
                    ffprobe_changed = (
                        self._timeout_capabilities.mark_ffprobe_timeout_unsupported()
                        or ffprobe_changed
                    )
        if ffmpeg_changed:
            logger.warning(
                "ffmpeg timeout options unsupported; disabling RTSP timeout options for this process"
            )
        if ffprobe_changed:
            logger.warning(
                "ffprobe timeout options unsupported; disabling RTSP timeout options for this process"
            )

    def _wait_for_process_exit(
        self,
        proc: subprocess.Popen[str] | None,
        *,
        timeout_s: float,
    ) -> tuple[int | None, str]:
        if proc is None:
            return None, ""

        try:
            _, stderr_text = proc.communicate(timeout=max(0.1, timeout_s))
            stderr_tail = stderr_text[-240:] if stderr_text else ""
            return proc.poll(), stderr_tail.strip()
        except Exception:
            return self._terminate_process(proc)

    def _terminate_process(self, proc: subprocess.Popen[str] | None) -> tuple[int | None, str]:
        if proc is None:
            return None, ""

        stderr_tail = ""
        try:
            if proc.poll() is None:
                if not _signal_process_group(proc.pid, signal.SIGTERM):
                    proc.terminate()
            _, stderr_text = proc.communicate(timeout=2)
            stderr_tail = stderr_text[-240:] if stderr_text else ""
        except Exception:
            try:
                if not _signal_process_group(proc.pid, signal.SIGKILL):
                    proc.kill()
                _, stderr_text = proc.communicate(timeout=2)
                stderr_tail = stderr_text[-240:] if stderr_text else ""
            except Exception:
                return proc.poll(), ""

        return proc.poll(), stderr_tail.strip()


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


def _format_recording_cmd(cmd: list[str]) -> str:
    safe_cmd = list(cmd)
    try:
        idx = safe_cmd.index("-i")
        safe_cmd[idx + 1] = _redact_rtsp_url(str(safe_cmd[idx + 1]))
    except Exception:
        pass
    return _format_cmd(safe_cmd)


def _sanitize_stderr(stderr_text: str, input_url: str) -> str:
    if not stderr_text:
        return ""
    return stderr_text.replace(input_url, _redact_rtsp_url(input_url)).strip()


def _is_session_limit_error(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(hint in lowered for hint in _SESSION_LIMIT_HINTS)


def _collect_recording_stability_signals(stderr_text: str) -> RecordingValidationSignals:
    if not stderr_text:
        return RecordingValidationSignals()

    def _count(pattern: str) -> int:
        return len(re.findall(pattern, stderr_text, flags=re.IGNORECASE))

    return RecordingValidationSignals(
        non_monotonic_dts=_count(r"Non-monotonic DTS"),
        queue_input_backward=_count(r"Queue input is backward in time"),
        dts_discontinuity=_count(r"DTS discontinuity"),
        sei_truncated=_count(r"SEI type 764 size"),
    )
