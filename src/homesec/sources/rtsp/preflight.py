from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, Protocol

from pydantic import BaseModel, Field

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
from homesec.sources.rtsp.utils import _format_cmd, _is_timeout_option_error, _redact_rtsp_url

logger = logging.getLogger(__name__)

_NON_MONOTONIC_DTS_RETRY_THRESHOLD = 5
_QUEUE_BACKWARD_RETRY_THRESHOLD = 1
_DTS_DISCONTINUITY_RETRY_THRESHOLD = 1


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
    negotiation_attempts: list[str] = Field(default_factory=list)
    selected_recording_profile: str | None = None
    session_mode: Literal["dual_stream", "single_stream"] | None = None
    notes: list[str] = Field(default_factory=list)


class CameraPreflightOutcome(BaseModel):
    """Locked startup preflight result for one RTSP camera."""

    model_config = {"extra": "forbid"}

    camera_key: str
    motion_profile: MotionProfile
    recording_profile: RecordingProfile
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
    ) -> None:
        self._output_dir = output_dir
        self._rtsp_connect_timeout_s = rtsp_connect_timeout_s
        self._rtsp_io_timeout_s = rtsp_io_timeout_s
        self._session_overlap_s = max(0.1, session_overlap_s)
        self._command_timeout_s = max(1.0, command_timeout_s)
        self._discovery = discovery or FfprobeStreamDiscovery(
            rtsp_connect_timeout_s=rtsp_connect_timeout_s,
            rtsp_io_timeout_s=rtsp_io_timeout_s,
            command_timeout_s=command_timeout_s,
        )

    def run(
        self,
        *,
        camera_name: str,
        primary_rtsp_url: str,
        detect_rtsp_url: str,
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
            return CameraPreflightOutcome(
                camera_key=camera_key,
                motion_profile=motion_profile,
                recording_profile=recording_profile,
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
        return CameraPreflightOutcome(
            camera_key=camera_key,
            motion_profile=motion_profile,
            recording_profile=recording_profile,
            diagnostics=diagnostics,
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

        attempts: list[tuple[str, list[str]]] = []
        if timeout_args:
            attempts.append(("timeouts", timeout_args))
        attempts.append(("no_timeouts" if timeout_args else "default", []))

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
                    try:
                        if temp_path.exists() and temp_path.stat().st_size > 0:
                            return RecordingValidationResult(ok=True, signals=signals)
                    except Exception:
                        return RecordingValidationResult(ok=True, signals=signals)
                    return RecordingValidationResult(
                        ok=False, error="ffmpeg returned success but output clip is empty"
                    )

                if label == "timeouts" and _is_timeout_option_error(stderr_text):
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
        timeout_args = self._timeout_args()
        attempts: list[tuple[str, list[str]]] = []
        if timeout_args:
            attempts.append(("timeouts", timeout_args))
        attempts.append(("no_timeouts" if timeout_args else "default", []))

        for label, current_timeout_args in attempts:
            motion_cmd = self._build_stream_open_cmd(
                input_url=motion_url,
                timeout_args=current_timeout_args,
            )
            recording_cmd = self._build_stream_open_cmd(
                input_url=recording_url,
                timeout_args=current_timeout_args,
            )

            motion_proc: subprocess.Popen[str] | None = None
            recording_proc: subprocess.Popen[str] | None = None

            try:
                motion_proc = subprocess.Popen(
                    motion_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                recording_proc = subprocess.Popen(
                    recording_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except Exception as exc:
                logger.warning(
                    "Session-limit validation failed to launch ffmpeg (%s): %s",
                    label,
                    exc,
                    exc_info=True,
                )
                self._terminate_process(motion_proc)
                self._terminate_process(recording_proc)
                continue

            time.sleep(self._session_overlap_s)

            motion_exit = motion_proc.poll()
            recording_exit = recording_proc.poll()
            if motion_exit is None and recording_exit is None:
                self._terminate_process(motion_proc)
                self._terminate_process(recording_proc)
                return True

            motion_err = self._terminate_process(motion_proc)
            recording_err = self._terminate_process(recording_proc)
            timeout_option_error = label == "timeouts" and (
                _is_timeout_option_error(motion_err) or _is_timeout_option_error(recording_err)
            )
            if timeout_option_error:
                logger.debug(
                    "Session-limit validation retrying without timeout options; motion_err=%s recording_err=%s",
                    motion_err,
                    recording_err,
                )
                continue

            logger.warning(
                "Session-limit validation failed (%s): motion_exit=%s recording_exit=%s motion_err=%s recording_err=%s",
                label,
                motion_exit,
                recording_exit,
                motion_err,
                recording_err,
            )

        return False

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
            "error",
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

    def _build_stream_open_cmd(self, *, input_url: str, timeout_args: list[str]) -> list[str]:
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
                "2",
                "-an",
                "-f",
                "null",
                "-",
            ]
        )
        return cmd

    def _timeout_args(self) -> list[str]:
        timeout_args: list[str] = []
        if self._rtsp_connect_timeout_s > 0:
            timeout_us_connect = str(int(max(0.1, self._rtsp_connect_timeout_s) * 1_000_000))
            timeout_args.extend(["-stimeout", timeout_us_connect])
        if self._rtsp_io_timeout_s > 0:
            timeout_us_io = str(int(max(0.1, self._rtsp_io_timeout_s) * 1_000_000))
            timeout_args.extend(["-rw_timeout", timeout_us_io])
        return timeout_args

    def _terminate_process(self, proc: subprocess.Popen[str] | None) -> str:
        if proc is None:
            return ""

        stderr_tail = ""
        try:
            if proc.poll() is None:
                proc.terminate()
            _, stderr_text = proc.communicate(timeout=2)
            stderr_tail = stderr_text[-240:] if stderr_text else ""
        except Exception:
            try:
                proc.kill()
                _, stderr_text = proc.communicate(timeout=2)
                stderr_tail = stderr_text[-240:] if stderr_text else ""
            except Exception:
                return ""

        return stderr_tail.strip()


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
