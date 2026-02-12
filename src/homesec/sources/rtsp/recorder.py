from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Protocol

from homesec.sources.rtsp.capabilities import (
    RTSPTimeoutCapabilities,
    get_global_rtsp_timeout_capabilities,
)
from homesec.sources.rtsp.clock import Clock
from homesec.sources.rtsp.recording_profile import RecordingProfile, build_default_recording_profile
from homesec.sources.rtsp.utils import (
    _build_timeout_attempts,
    _format_cmd,
    _is_timeout_option_error,
    _redact_rtsp_url,
)

logger = logging.getLogger(__name__)


class Recorder(Protocol):
    def start(self, output_file: Path, stderr_log: Path) -> subprocess.Popen[bytes] | None: ...

    def stop(self, proc: subprocess.Popen[bytes], output_file: Path | None) -> None: ...

    def is_alive(self, proc: subprocess.Popen[bytes]) -> bool: ...


class FfmpegRecorder:
    def __init__(
        self,
        *,
        rtsp_url: str,
        ffmpeg_flags: list[str],
        rtsp_connect_timeout_s: float,
        rtsp_io_timeout_s: float,
        clock: Clock,
        recording_profile: RecordingProfile | None = None,
        timeout_capabilities: RTSPTimeoutCapabilities | None = None,
    ) -> None:
        self._rtsp_url = rtsp_url
        self._ffmpeg_flags = ffmpeg_flags
        self._recording_profile = recording_profile or build_default_recording_profile(rtsp_url)
        self._rtsp_connect_timeout_s = rtsp_connect_timeout_s
        self._rtsp_io_timeout_s = rtsp_io_timeout_s
        self._clock = clock
        self._timeout_capabilities = timeout_capabilities or get_global_rtsp_timeout_capabilities()

    def configure_profile(self, profile: RecordingProfile) -> None:
        self._recording_profile = profile

    def start(self, output_file: Path, stderr_log: Path) -> subprocess.Popen[bytes] | None:
        def _read_tail(path: Path, max_bytes: int = 4000) -> str:
            try:
                data = path.read_bytes()
            except Exception as exc:
                logger.warning("Failed to read recording stderr tail: %s", exc, exc_info=True)
                return ""
            if len(data) <= max_bytes:
                return data.decode(errors="replace")
            return data[-max_bytes:].decode(errors="replace")

        cmd_base = [
            "ffmpeg",
            "-rtsp_transport",
            "tcp",
            "-rtsp_flags",
            "prefer_tcp",
            "-user_agent",
            "Lavf",
        ]

        user_flags = self._ffmpeg_flags
        timeout_args = self._timeout_capabilities.build_ffmpeg_timeout_args_for_user_flags(
            connect_timeout_s=self._rtsp_connect_timeout_s,
            io_timeout_s=self._rtsp_io_timeout_s,
            user_flags=user_flags,
        )

        input_url = self._recording_profile.input_url
        cmd_tail = list(self._recording_profile.ffmpeg_input_args)
        cmd_tail.extend(["-i", input_url])

        # Naive check to see if user overrode defaults
        # If user supplies ANY -loglevel, we don't add ours.
        # If user supplies ANY -fflags, we don't add ours (to avoid concatenation complexity).
        # This allows full user control.
        has_loglevel = any(x == "-loglevel" for x in user_flags)
        if not has_loglevel:
            cmd_tail.extend(["-loglevel", "warning"])

        has_fflags = any(x == "-fflags" for x in user_flags)
        if not has_fflags:
            cmd_tail.extend(["-fflags", "+genpts+igndts"])

        cmd_tail.extend(self._recording_profile.ffmpeg_output_args)

        # Add user flags last so they can potentially override or add to the above
        cmd_tail.extend(user_flags)
        cmd_tail.extend(["-y", str(output_file)])

        attempts = _build_timeout_attempts(timeout_args)

        for label, extra_args in attempts:
            cmd = list(cmd_base) + list(extra_args) + cmd_tail

            safe_cmd = list(cmd)
            try:
                idx = safe_cmd.index("-i")
                safe_cmd[idx + 1] = _redact_rtsp_url(str(safe_cmd[idx + 1]))
            except Exception as exc:
                logger.warning("Failed to redact recording RTSP URL: %s", exc, exc_info=True)
            logger.debug("Recording ffmpeg (%s): %s", label, _format_cmd(safe_cmd))

            try:
                with open(stderr_log, "w") as stderr_file:
                    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr_file)

                self._clock.sleep(0.5)
                if proc.poll() is None:
                    if label == "timeouts":
                        self._timeout_capabilities.note_ffmpeg_timeout_success()
                    return proc

                stderr_tail = _read_tail(stderr_log)
                timeout_option_error = (
                    label == "timeouts"
                    and bool(stderr_tail)
                    and _is_timeout_option_error(stderr_tail)
                )

                if timeout_option_error:
                    changed = self._timeout_capabilities.mark_ffmpeg_timeout_unsupported()
                    if changed:
                        logger.warning(
                            "Recording process died immediately (%s, exit code: %s); timeout options unsupported. Disabling RTSP timeout options for this process",
                            label,
                            proc.returncode,
                        )
                    else:
                        logger.debug(
                            "Recording timeout options already disabled after repeated failure (%s, exit code: %s)",
                            label,
                            proc.returncode,
                        )
                    check_logs_fn = logger.warning if changed else logger.debug
                    check_logs_fn("Check logs at: %s", stderr_log)
                else:
                    logger.error(
                        "Recording process died immediately (%s, exit code: %s)",
                        label,
                        proc.returncode,
                    )
                    logger.error("Check logs at: %s", stderr_log)

                if stderr_tail:
                    redacted_tail = stderr_tail.replace(input_url, _redact_rtsp_url(input_url))
                    if timeout_option_error:
                        log_fn = logger.warning if changed else logger.debug
                        log_fn("Recording stderr tail (%s):\n%s", label, redacted_tail)
                        logger.warning(
                            "Recording ffmpeg missing timeout options; retrying without timeouts"
                        )
                        continue
                    logger.error("Recording stderr tail (%s):\n%s", label, redacted_tail)
                if label == "timeouts":
                    return None
            except Exception:
                logger.exception("Failed to start recording")
                return None

        return None

    def stop(self, proc: subprocess.Popen[bytes], output_file: Path | None) -> None:
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Recording process did not terminate, killing (PID: %s)", proc.pid)
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                logger.exception("Failed to kill recording process (PID: %s)", proc.pid)
        except Exception:
            logger.exception("Failed while stopping recording process (PID: %s)", proc.pid)

        logger.debug(
            "Stopped recording: %s",
            output_file,
            extra={"recording_id": output_file.name if output_file else None},
        )

    def is_alive(self, proc: subprocess.Popen[bytes]) -> bool:
        return proc.poll() is None
