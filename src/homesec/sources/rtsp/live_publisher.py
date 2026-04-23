from __future__ import annotations

import logging
import shutil
import signal
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Event, RLock, Thread, current_thread
from typing import Literal, Protocol, TextIO

from homesec.sources.rtsp.capabilities import (
    RTSPTimeoutCapabilities,
    get_global_rtsp_timeout_capabilities,
)
from homesec.sources.rtsp.clock import Clock, SystemClock
from homesec.sources.rtsp.discovery import (
    CameraProbeResult,
    FfprobeStreamDiscovery,
    ProbeError,
    ProbeStreamInfo,
    build_camera_key,
)
from homesec.sources.rtsp.utils import (
    _build_timeout_attempts,
    _format_cmd,
    _is_timeout_option_error,
    _redact_rtsp_url,
    _signal_process_group,
)

logger = logging.getLogger(__name__)

_READY_POLL_INTERVAL_S = 0.1
_DEFAULT_VIEWER_WINDOW_S = 2.0
_START_FAILURE_MAX_BYTES = 4_000
_HLS_BROWSER_COPY_AUDIO_CODECS: frozenset[str] = frozenset({"aac"})
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


class LivePublisherState(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    ERROR = "error"


class LivePublisherRefusalReason(StrEnum):
    RECORDING_PRIORITY = "recording_priority"
    SESSION_BUDGET_EXHAUSTED = "session_budget_exhausted"
    PREVIEW_TEMPORARILY_UNAVAILABLE = "preview_temporarily_unavailable"


@dataclass(frozen=True, slots=True)
class LivePublisherStatus:
    state: LivePublisherState
    viewer_count: int | None = None
    degraded_reason: str | None = None
    last_error: str | None = None
    idle_shutdown_at: float | None = None


@dataclass(frozen=True, slots=True)
class LivePublisherStartRefusal:
    reason: LivePublisherRefusalReason
    message: str


class LivePublisher(Protocol):
    def status(self) -> LivePublisherStatus: ...

    def ensure_active(self) -> LivePublisherStatus | LivePublisherStartRefusal: ...

    def request_stop(self) -> None: ...

    def note_viewer_activity(self, viewer_id: str | None = None) -> None: ...

    def sync_recording_active(self, recording_active: bool) -> None: ...

    def shutdown(self) -> None: ...


class StreamDiscovery(Protocol):
    def probe(
        self,
        *,
        camera_key: str,
        candidate_urls: list[str],
    ) -> CameraProbeResult | ProbeError: ...


@dataclass(frozen=True, slots=True)
class _CodecPlan:
    video_args: tuple[str, ...]
    audio_args: tuple[str, ...]


class HLSLivePublisher(LivePublisher):
    """Runtime-owned RTSP preview publisher that writes a live HLS window."""

    def __init__(
        self,
        *,
        camera_name: str,
        rtsp_url: str,
        storage_dir: Path,
        segment_duration_ms: int,
        live_window_segments: int,
        idle_timeout_s: float,
        audio_enabled: bool,
        audio_codec: Literal["auto", "copy", "aac"] = "auto",
        video_codec: Literal["auto", "copy", "h264"] = "auto",
        rtsp_connect_timeout_s: float,
        rtsp_io_timeout_s: float,
        clock: Clock | None = None,
        timeout_capabilities: RTSPTimeoutCapabilities | None = None,
        stream_discovery: StreamDiscovery | None = None,
        background_maintenance: bool = True,
        maintenance_interval_s: float | None = None,
    ) -> None:
        self._camera_name = camera_name
        self._camera_slug = _sanitize_camera_name(camera_name)
        self._camera_key = build_camera_key(camera_name, rtsp_url)
        self._rtsp_url = rtsp_url
        self._storage_dir = Path(storage_dir)
        self._camera_dir = self._storage_dir / "homesec" / self._camera_slug
        self._playlist_path = self._camera_dir / "playlist.m3u8"
        self._stderr_log_path = self._camera_dir / "preview_ffmpeg.log"
        self._segment_filename_pattern = self._camera_dir / "segment_%06d.ts"
        self._segment_duration_s = max(0.001, float(segment_duration_ms) / 1000.0)
        self._live_window_segments = int(live_window_segments)
        self._idle_timeout_s = max(0.0, float(idle_timeout_s))
        self._audio_enabled = bool(audio_enabled)
        self._audio_codec = audio_codec
        self._video_codec = video_codec
        self._rtsp_connect_timeout_s = float(rtsp_connect_timeout_s)
        self._rtsp_io_timeout_s = float(rtsp_io_timeout_s)
        self._clock = clock or SystemClock()
        self._timeout_capabilities = timeout_capabilities or get_global_rtsp_timeout_capabilities()
        self._stream_discovery = stream_discovery or FfprobeStreamDiscovery(
            rtsp_connect_timeout_s=self._rtsp_connect_timeout_s,
            rtsp_io_timeout_s=self._rtsp_io_timeout_s,
            timeout_capabilities=self._timeout_capabilities,
        )
        self._background_maintenance = background_maintenance
        self._maintenance_interval_s = (
            float(maintenance_interval_s)
            if maintenance_interval_s is not None
            else min(max(self._segment_duration_s, 0.25), 1.0)
        )
        self._viewer_window_s = max(_DEFAULT_VIEWER_WINDOW_S, self._segment_duration_s * 2.0)
        self._startup_timeout_s = max(3.0, min(10.0, (self._segment_duration_s * 4.0) + 2.0))

        self._lock = RLock()
        self._maintenance_stop = Event()
        self._maintenance_thread: Thread | None = None
        self._status = LivePublisherStatus(state=LivePublisherState.IDLE, viewer_count=0)
        self._last_cancellation_result: LivePublisherStatus | LivePublisherStartRefusal = (
            self._status
        )
        self._completed_start_results: dict[
            int, LivePublisherStatus | LivePublisherStartRefusal
        ] = {}
        self._queued_start_waiters: dict[int, int] = {}
        self._start_in_progress = False
        self._active_start_token = 0
        self._cancelled_start_token = 0
        self._stop_request_token = 0
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_handle: TextIO | None = None
        self._recording_active = False
        self._viewer_activity: dict[str, float] = {}
        self._anonymous_viewer_last_seen: float | None = None
        self._last_activity_at: float | None = None

    def status(self) -> LivePublisherStatus:
        with self._lock:
            self._refresh_locked(now=self._clock.now())
            return self._status

    def ensure_active(self) -> LivePublisherStatus | LivePublisherStartRefusal:
        start_token = 0
        queued_start_token = 0
        stop_request_token = self._stop_request_token
        while True:
            with self._lock:
                now = self._clock.now()
                self._refresh_locked(now=now)
                if queued_start_token and self._start_was_cancelled_locked(queued_start_token):
                    self._release_queued_start_waiter_locked(queued_start_token)
                    return self._last_cancellation_result
                if self._stop_requested_since_locked(stop_request_token):
                    self._release_queued_start_waiter_locked(queued_start_token)
                    return self._last_cancellation_result
                if self._recording_active:
                    self._release_queued_start_waiter_locked(queued_start_token)
                    logger.info(
                        "Refusing preview activation because recording is active: camera=%s",
                        self._camera_name,
                    )
                    return self._recording_priority_refusal()

                completed_start_result = self._completed_start_results.get(queued_start_token)
                if completed_start_result is not None:
                    # A queued caller may observe a completed READY result even if the
                    # underlying ffmpeg process has already exited. In that case, drop the
                    # cached READY result and proceed with normal startup logic.
                    if (
                        isinstance(completed_start_result, LivePublisherStatus)
                        and completed_start_result.state is LivePublisherState.READY
                        and not self._is_process_running_locked()
                    ):
                        self._completed_start_results.pop(queued_start_token, None)
                    else:
                        self._release_queued_start_waiter_locked(queued_start_token)
                        return completed_start_result

                if self._start_in_progress:
                    next_queued_start_token = self._active_start_token
                    if queued_start_token != next_queued_start_token:
                        self._release_queued_start_waiter_locked(queued_start_token)
                        queued_start_token = next_queued_start_token
                        self._register_queued_start_waiter_locked(queued_start_token)
                    self._sleep_without_lock_locked(_READY_POLL_INTERVAL_S)
                    continue

                if self._is_process_running_locked():
                    self._release_queued_start_waiter_locked(queued_start_token)
                    self._mark_activity_locked(now=now)
                    self._status = self._build_running_status_locked(
                        now=now,
                        state=LivePublisherState.READY,
                        degraded_reason=None,
                    )
                    self._ensure_maintenance_thread_started_locked()
                    return self._status

                self._release_queued_start_waiter_locked(queued_start_token)
                queued_start_token = 0
                self._start_in_progress = True
                self._active_start_token += 1
                start_token = self._active_start_token
                break

        start_result: LivePublisherStatus | LivePublisherStartRefusal | None = None
        try:
            start_result = self._start(start_token=start_token)
            return start_result
        finally:
            with self._lock:
                self._start_in_progress = False
                if start_result is not None and self._queued_start_waiters.get(start_token, 0) > 0:
                    self._completed_start_results[start_token] = start_result

    def request_stop(self) -> None:
        with self._lock:
            self._stop_request_token += 1
            self._cancel_pending_start_locked()
            self._stop_locked(clear_error=True)
            self._last_cancellation_result = self._status

    def note_viewer_activity(self, viewer_id: str | None = None) -> None:
        with self._lock:
            now = self._clock.now()
            self._refresh_locked(now=now)
            if not self._is_process_running_locked():
                return

            self._mark_activity_locked(now=now)
            if viewer_id is None:
                self._anonymous_viewer_last_seen = now
            else:
                self._viewer_activity[viewer_id] = now
            self._status = self._build_running_status_locked(
                now=now,
                state=LivePublisherState.READY,
                degraded_reason=None,
            )

    def sync_recording_active(self, recording_active: bool) -> None:
        with self._lock:
            was_recording_active = self._recording_active
            had_active_preview = self._is_process_running_locked()
            had_pending_start = self._start_in_progress
            self._recording_active = recording_active
            if not recording_active:
                self._refresh_locked(now=self._clock.now())
                return

            if not was_recording_active:
                if had_active_preview:
                    logger.info(
                        "Stopping preview because recording became active: camera=%s",
                        self._camera_name,
                    )
                elif had_pending_start:
                    logger.info(
                        "Cancelling preview startup because recording became active: camera=%s",
                        self._camera_name,
                    )
                else:
                    logger.info(
                        "Recording became active; preview startup will be refused: camera=%s",
                        self._camera_name,
                    )

            self._cancel_pending_start_locked()
            self._stop_locked(clear_error=True)
            self._last_cancellation_result = self._recording_priority_refusal()

    def shutdown(self) -> None:
        maintenance_thread: Thread | None = None
        with self._lock:
            self._stop_request_token += 1
            self._cancel_pending_start_locked()
            self._stop_locked(clear_error=True)
            self._last_cancellation_result = self._status
            self._maintenance_stop.set()
            maintenance_thread = self._maintenance_thread
            self._maintenance_thread = None

        if maintenance_thread is not None and maintenance_thread is not current_thread():
            maintenance_thread.join(timeout=2.0)

    def _start(
        self,
        *,
        start_token: int,
    ) -> LivePublisherStatus | LivePublisherStartRefusal:
        codec_plan = self._build_codec_plan()
        timeout_args = self._timeout_capabilities.build_ffmpeg_timeout_args_for_user_flags(
            connect_timeout_s=self._rtsp_connect_timeout_s,
            io_timeout_s=self._rtsp_io_timeout_s,
            user_flags=[],
        )

        attempts = _build_timeout_attempts(timeout_args)
        last_refusal: LivePublisherStartRefusal | None = None

        for label, timeout_attempt_args in attempts:
            attempt_refusal: LivePublisherStartRefusal | None = None
            timeout_option_error = False
            stop_error: str | None = None
            with self._lock:
                if self._start_was_cancelled_locked(start_token):
                    self._stop_locked(clear_error=True)
                    return self._last_cancellation_result

                if self._recording_active:
                    return self._recording_priority_refusal()

                if not self._prepare_live_dir_locked():
                    return self._set_error_locked(
                        reason=LivePublisherRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE,
                        message="Preview storage directory is unavailable",
                        last_error="Preview storage directory could not be prepared",
                    )

                cmd = self._build_ffmpeg_cmd(
                    codec_plan=codec_plan,
                    timeout_args=timeout_attempt_args,
                )
                self._log_ffmpeg_cmd(label=label, cmd=cmd)

                try:
                    stderr_handle = open(self._stderr_log_path, "w", encoding="utf-8")
                except Exception as exc:
                    logger.warning("Failed to open preview stderr log: %s", exc, exc_info=True)
                    return self._set_error_locked(
                        reason=LivePublisherRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE,
                        message="Preview stderr log could not be created",
                        last_error=f"Preview stderr log create failed: {type(exc).__name__}",
                    )

                try:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=stderr_handle,
                        start_new_session=True,
                    )
                except Exception as exc:
                    stderr_handle.close()
                    logger.warning("Failed to start preview ffmpeg: %s", exc, exc_info=True)
                    return self._set_error_locked(
                        reason=LivePublisherRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE,
                        message="Preview ffmpeg could not be started",
                        last_error=f"Preview ffmpeg start failed: {type(exc).__name__}",
                    )

                self._process = process
                self._stderr_handle = stderr_handle
                started_now = self._clock.now()
                self._mark_activity_locked(now=started_now)
                self._status = LivePublisherStatus(
                    state=LivePublisherState.STARTING,
                    viewer_count=self._viewer_count_locked(now=started_now),
                    idle_shutdown_at=started_now + self._idle_timeout_s,
                )

                if self._wait_until_ready_locked():
                    if label == "timeouts":
                        self._timeout_capabilities.note_ffmpeg_timeout_success()
                    ready_now = self._clock.now()
                    self._mark_activity_locked(now=ready_now)
                    self._ensure_maintenance_thread_started_locked()
                    self._status = self._build_running_status_locked(
                        now=ready_now,
                        state=LivePublisherState.READY,
                        degraded_reason=None,
                    )
                    return self._status

                if self._start_was_cancelled_locked(start_token):
                    self._stop_locked(clear_error=True)
                    return self._last_cancellation_result

                if self._recording_active:
                    self._stop_locked(clear_error=True)
                    return self._recording_priority_refusal()

                stderr_tail = self._read_stderr_tail_locked()
                timeout_option_error = (
                    bool(stderr_tail)
                    and label == "timeouts"
                    and (_is_timeout_option_error(stderr_tail))
                )
                stop_error = self._stop_locked(clear_error=False)

                if stop_error is not None:
                    stderr_tail = f"{stderr_tail}; {stop_error}" if stderr_tail else stop_error

                if not timeout_option_error or stop_error is not None:
                    refusal_reason = _classify_start_refusal(stderr_tail)
                    message = (
                        "Preview could not start because the camera session budget is exhausted"
                        if refusal_reason is LivePublisherRefusalReason.SESSION_BUDGET_EXHAUSTED
                        else "Preview publisher failed to become ready"
                    )
                    attempt_refusal = self._set_error_locked(
                        reason=refusal_reason,
                        message=message,
                        last_error=stderr_tail
                        or "Preview publisher exited before producing HLS output",
                    )

            if timeout_option_error and stop_error is None:
                changed = self._timeout_capabilities.mark_ffmpeg_timeout_unsupported()
                if changed:
                    logger.warning(
                        "Preview ffmpeg timeout options unsupported; retrying without timeout options"
                    )
                continue

            if attempt_refusal is not None:
                last_refusal = attempt_refusal
                break

        if last_refusal is not None:
            return last_refusal

        with self._lock:
            return self._set_error_locked(
                reason=LivePublisherRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE,
                message="Preview publisher failed to start",
                last_error="Preview publisher failed after retrying timeout options",
            )

    def _wait_until_ready_locked(self) -> bool:
        deadline = self._clock.now() + self._startup_timeout_s
        while self._clock.now() < deadline:
            if not self._is_process_running_locked():
                return False
            if self._output_ready_locked():
                return self._is_process_running_locked()
            self._sleep_without_lock_locked(_READY_POLL_INTERVAL_S)
        if not self._output_ready_locked():
            return False
        return self._is_process_running_locked()

    def _refresh_locked(self, *, now: float) -> None:
        self._expire_viewers_locked(now=now)

        if self._process is not None and not self._is_process_running_locked():
            exit_code = self._process.poll()
            stderr_tail = self._read_stderr_tail_locked()
            self._process = None
            self._close_process_handles_locked()
            self._cleanup_live_dir_locked()
            self._viewer_activity.clear()
            self._anonymous_viewer_last_seen = None
            self._last_activity_at = None
            self._status = LivePublisherStatus(
                state=LivePublisherState.ERROR,
                viewer_count=0,
                last_error=stderr_tail or f"Preview ffmpeg exited unexpectedly ({exit_code})",
            )
            return

        if not self._is_process_running_locked():
            if self._status.state not in (LivePublisherState.ERROR,):
                self._status = LivePublisherStatus(state=LivePublisherState.IDLE, viewer_count=0)
            return

        viewer_count = self._viewer_count_locked(now=now)
        if viewer_count == 0 and self._last_activity_at is not None:
            idle_shutdown_at = self._last_activity_at + self._idle_timeout_s
            if now >= idle_shutdown_at:
                logger.info(
                    "Stopping preview publisher after idle timeout: camera=%s",
                    self._camera_name,
                )
                self._stop_locked(clear_error=True)
                return

        self._status = self._build_running_status_locked(
            now=now,
            state=LivePublisherState.READY,
            degraded_reason=None,
        )

    def _build_codec_plan(self) -> _CodecPlan:
        stream_info = self._probe_stream_info()
        video_args: tuple[str, ...]
        audio_args: tuple[str, ...]

        if self._video_codec == "copy":
            video_args = ("-c:v", "copy")
        elif self._video_codec == "h264":
            video_args = _h264_transcode_args(self._segment_duration_s)
        elif stream_info is not None and (stream_info.video_codec or "").lower() == "h264":
            video_args = ("-c:v", "copy")
        else:
            video_args = _h264_transcode_args(self._segment_duration_s)

        if not self._audio_enabled:
            audio_args = ("-an",)
        elif self._audio_codec == "copy":
            audio_args = ("-map", "0:a:0?", "-c:a", "copy")
        elif self._audio_codec == "aac":
            audio_args = ("-map", "0:a:0?", "-c:a", "aac", "-b:a", "128k")
        elif stream_info is not None and _is_hls_browser_audio_copy_compatible(
            stream_info.audio_codec
        ):
            audio_args = ("-map", "0:a:0?", "-c:a", "copy")
        else:
            audio_args = ("-map", "0:a:0?", "-c:a", "aac", "-b:a", "128k")

        return _CodecPlan(video_args=video_args, audio_args=audio_args)

    def _probe_stream_info(self) -> ProbeStreamInfo | None:
        needs_probe = self._video_codec == "auto" or (
            self._audio_enabled and self._audio_codec == "auto"
        )
        if not needs_probe:
            return None

        result = self._stream_discovery.probe(
            camera_key=self._camera_key,
            candidate_urls=[self._rtsp_url],
        )
        match result:
            case ProbeError() as err:
                logger.warning(
                    "Preview stream probe failed for %s: %s",
                    self._camera_name,
                    err.message,
                )
                return None
            case CameraProbeResult() as probe_result:
                for stream in probe_result.streams:
                    if stream.probe_ok:
                        return stream
                return None
            case _:
                logger.warning(
                    "Preview stream probe returned unexpected type for %s: %s",
                    self._camera_name,
                    type(result).__name__,
                )
                return None

    def _build_ffmpeg_cmd(
        self,
        *,
        codec_plan: _CodecPlan,
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
            "-rtsp_flags",
            "prefer_tcp",
            "-user_agent",
            "Lavf",
        ]
        cmd.extend(timeout_args)
        cmd.extend(["-fflags", "+genpts+igndts", "-i", self._rtsp_url, "-map", "0:v:0"])
        cmd.extend(codec_plan.video_args)
        cmd.extend(codec_plan.audio_args)
        cmd.extend(
            [
                "-muxdelay",
                "0",
                "-muxpreload",
                "0",
                "-f",
                "hls",
                "-hls_time",
                _format_segment_duration(self._segment_duration_s),
                "-hls_list_size",
                str(self._live_window_segments),
                "-hls_delete_threshold",
                "1",
                "-hls_allow_cache",
                "0",
                "-hls_flags",
                "delete_segments+append_list+omit_endlist+program_date_time+temp_file",
                "-hls_segment_filename",
                str(self._segment_filename_pattern),
                "-start_number",
                "0",
                "-y",
                str(self._playlist_path),
            ]
        )
        return cmd

    def _build_running_status_locked(
        self,
        *,
        now: float,
        state: LivePublisherState,
        degraded_reason: str | None,
    ) -> LivePublisherStatus:
        viewer_count = self._viewer_count_locked(now=now)
        idle_shutdown_at: float | None = None
        if viewer_count == 0 and self._last_activity_at is not None:
            idle_shutdown_at = self._last_activity_at + self._idle_timeout_s
        return LivePublisherStatus(
            state=state,
            viewer_count=viewer_count,
            degraded_reason=degraded_reason,
            last_error=None,
            idle_shutdown_at=idle_shutdown_at,
        )

    def _output_ready_locked(self) -> bool:
        if not self._playlist_path.exists():
            return False
        try:
            playlist_text = self._playlist_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read preview playlist: %s", exc, exc_info=True)
            return False

        if not playlist_text.strip():
            return False

        for line in playlist_text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return True

        return any(self._camera_dir.glob("segment_*.ts"))

    def _stop_locked(self, *, clear_error: bool) -> str | None:
        has_live_window = self._camera_dir.exists()
        if self._process is None and self._stderr_handle is None and not has_live_window:
            if clear_error:
                self._status = LivePublisherStatus(state=LivePublisherState.IDLE, viewer_count=0)
            return None

        self._status = LivePublisherStatus(state=LivePublisherState.STOPPING, viewer_count=0)
        teardown_errors: list[str] = []
        process = self._process
        self._process = None
        if process is not None:
            stop_error = self._terminate_process(process)
            if stop_error is not None:
                teardown_errors.append(stop_error)
        self._close_process_handles_locked()
        if not self._cleanup_live_dir_locked():
            teardown_errors.append("Preview live output could not be removed")
        self._viewer_activity.clear()
        self._anonymous_viewer_last_seen = None
        self._last_activity_at = None

        if clear_error:
            if teardown_errors:
                self._status = LivePublisherStatus(
                    state=LivePublisherState.ERROR,
                    viewer_count=0,
                    last_error="; ".join(teardown_errors),
                )
            else:
                self._status = LivePublisherStatus(state=LivePublisherState.IDLE, viewer_count=0)

        if teardown_errors:
            return "; ".join(teardown_errors)
        return None

    def _terminate_process(self, process: subprocess.Popen[bytes]) -> str | None:
        try:
            if process.poll() is None:
                if not _signal_process_group(process.pid, signal.SIGTERM):
                    process.terminate()
                process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Preview process did not terminate, killing: camera=%s pid=%s",
                self._camera_name,
                process.pid,
            )
            try:
                if not _signal_process_group(process.pid, signal.SIGKILL):
                    process.kill()
                process.wait(timeout=2.0)
            except Exception:
                logger.exception(
                    "Failed to kill preview process: camera=%s pid=%s",
                    self._camera_name,
                    process.pid,
                )
        except Exception:
            logger.exception(
                "Failed while stopping preview process: camera=%s pid=%s",
                self._camera_name,
                process.pid,
            )
        if process.poll() is None:
            return "Preview ffmpeg could not be stopped cleanly"
        return None

    def _prepare_live_dir_locked(self) -> bool:
        try:
            if not self._cleanup_live_dir_locked():
                return False
            self._camera_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            logger.exception(
                "Failed to prepare preview directory: camera=%s dir=%s",
                self._camera_name,
                self._camera_dir,
            )
            return False

    def _cleanup_live_dir_locked(self) -> bool:
        if not self._camera_dir.exists():
            return True
        try:
            shutil.rmtree(self._camera_dir)
            return True
        except Exception:
            logger.exception(
                "Failed to clean preview directory: camera=%s dir=%s",
                self._camera_name,
                self._camera_dir,
            )
            return False

    def _close_process_handles_locked(self) -> None:
        if self._stderr_handle is None:
            return
        try:
            self._stderr_handle.close()
        except Exception:
            logger.exception(
                "Failed to close preview stderr log: camera=%s path=%s",
                self._camera_name,
                self._stderr_log_path,
            )
        finally:
            self._stderr_handle = None

    def _cancel_pending_start_locked(self) -> None:
        if self._start_in_progress:
            self._cancelled_start_token = self._active_start_token

    def _register_queued_start_waiter_locked(self, start_token: int) -> None:
        if start_token <= 0:
            return
        self._queued_start_waiters[start_token] = self._queued_start_waiters.get(start_token, 0) + 1

    def _release_queued_start_waiter_locked(self, start_token: int) -> None:
        if start_token <= 0:
            return
        current_waiters = self._queued_start_waiters.get(start_token)
        if current_waiters is None:
            return
        if current_waiters <= 1:
            self._queued_start_waiters.pop(start_token, None)
            self._completed_start_results.pop(start_token, None)
            return
        self._queued_start_waiters[start_token] = current_waiters - 1

    def _start_was_cancelled_locked(self, start_token: int) -> bool:
        return self._cancelled_start_token >= start_token

    def _stop_requested_since_locked(self, stop_request_token: int) -> bool:
        return self._stop_request_token != stop_request_token

    def _sleep_without_lock_locked(self, seconds: float) -> None:
        self._lock.release()
        try:
            self._clock.sleep(seconds)
        finally:
            self._lock.acquire()

    def _mark_activity_locked(self, *, now: float) -> None:
        self._last_activity_at = now

    def _expire_viewers_locked(self, *, now: float) -> None:
        expired_viewers = [
            viewer_id
            for viewer_id, last_seen in self._viewer_activity.items()
            if (now - last_seen) > self._viewer_window_s
        ]
        for viewer_id in expired_viewers:
            self._viewer_activity.pop(viewer_id, None)

        if (
            self._anonymous_viewer_last_seen is not None
            and (now - self._anonymous_viewer_last_seen) > self._viewer_window_s
        ):
            self._anonymous_viewer_last_seen = None

    def _viewer_count_locked(self, *, now: float) -> int:
        self._expire_viewers_locked(now=now)
        count = len(self._viewer_activity)
        if self._anonymous_viewer_last_seen is not None:
            count += 1
        return count

    def _is_process_running_locked(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _read_stderr_tail_locked(self) -> str:
        if not self._stderr_log_path.exists():
            return ""
        try:
            data = self._stderr_log_path.read_bytes()
        except Exception as exc:
            logger.warning("Failed to read preview stderr tail: %s", exc, exc_info=True)
            return ""

        if len(data) > _START_FAILURE_MAX_BYTES:
            data = data[-_START_FAILURE_MAX_BYTES:]
        text = data.decode(errors="replace")
        return text.replace(self._rtsp_url, _redact_rtsp_url(self._rtsp_url)).strip()

    def _log_ffmpeg_cmd(self, *, label: str, cmd: list[str]) -> None:
        safe_cmd = list(cmd)
        try:
            input_idx = safe_cmd.index("-i")
            safe_cmd[input_idx + 1] = _redact_rtsp_url(str(safe_cmd[input_idx + 1]))
        except Exception as exc:
            logger.warning("Failed to redact preview RTSP URL: %s", exc, exc_info=True)
        logger.debug(
            "Preview ffmpeg (%s): %s",
            label,
            _format_cmd(safe_cmd),
        )

    def _set_error_locked(
        self,
        *,
        reason: LivePublisherRefusalReason,
        message: str,
        last_error: str,
    ) -> LivePublisherStartRefusal:
        self._status = LivePublisherStatus(
            state=LivePublisherState.ERROR,
            viewer_count=0,
            last_error=last_error,
        )
        return LivePublisherStartRefusal(reason=reason, message=message)

    def _recording_priority_refusal(self) -> LivePublisherStartRefusal:
        return LivePublisherStartRefusal(
            reason=LivePublisherRefusalReason.RECORDING_PRIORITY,
            message="Preview is unavailable while recording is active for this camera",
        )

    def _ensure_maintenance_thread_started_locked(self) -> None:
        if not self._background_maintenance:
            return
        thread = self._maintenance_thread
        if thread is not None and thread.is_alive():
            return
        self._maintenance_stop.clear()
        self._maintenance_thread = Thread(
            target=self._maintenance_loop,
            name=f"preview-{self._camera_slug}",
            daemon=True,
        )
        self._maintenance_thread.start()

    def _maintenance_loop(self) -> None:
        while not self._maintenance_stop.wait(self._maintenance_interval_s):
            with self._lock:
                self._refresh_locked(now=self._clock.now())
                if not self._is_process_running_locked():
                    self._maintenance_thread = None
                    return


class NoopLivePublisher(LivePublisher):
    """Default preview seam used until a real publisher backend is wired in."""

    def __init__(self) -> None:
        self._status = LivePublisherStatus(state=LivePublisherState.IDLE)

    def status(self) -> LivePublisherStatus:
        return self._status

    def ensure_active(self) -> LivePublisherStatus | LivePublisherStartRefusal:
        return LivePublisherStartRefusal(
            reason=LivePublisherRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE,
            message="Preview publisher is not configured for this RTSP source",
        )

    def request_stop(self) -> None:
        self._status = LivePublisherStatus(state=LivePublisherState.IDLE)

    def note_viewer_activity(self, viewer_id: str | None = None) -> None:
        _ = viewer_id

    def sync_recording_active(self, recording_active: bool) -> None:
        _ = recording_active

    def shutdown(self) -> None:
        self._status = LivePublisherStatus(state=LivePublisherState.IDLE)


def _format_segment_duration(segment_duration_s: float) -> str:
    return f"{segment_duration_s:.3f}".rstrip("0").rstrip(".")


def _sanitize_camera_name(name: str) -> str:
    raw = str(name).strip()
    if not raw:
        return "camera"
    out: list[str] = []
    for ch in raw:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "camera"


def _classify_start_refusal(stderr_tail: str) -> LivePublisherRefusalReason:
    lowered = stderr_tail.lower()
    for hint in _SESSION_LIMIT_HINTS:
        if hint in lowered:
            return LivePublisherRefusalReason.SESSION_BUDGET_EXHAUSTED
    return LivePublisherRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE


def _is_hls_browser_audio_copy_compatible(audio_codec: str | None) -> bool:
    if audio_codec is None:
        return False
    return audio_codec.strip().lower() in _HLS_BROWSER_COPY_AUDIO_CODECS


def _h264_transcode_args(segment_duration_s: float) -> tuple[str, ...]:
    return (
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-sc_threshold",
        "0",
        "-force_key_frames",
        f"expr:gte(t,n_forced*{_format_segment_duration(segment_duration_s)})",
    )
