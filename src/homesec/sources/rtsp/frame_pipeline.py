from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any, Protocol

from homesec.sources.rtsp.clock import Clock
from homesec.sources.rtsp.hardware import HardwareAccelConfig
from homesec.sources.rtsp.utils import _format_cmd, _is_timeout_option_error, _redact_rtsp_url

logger = logging.getLogger(__name__)


class FramePipeline(Protocol):
    """Decode RTSP frames for motion detection.

    Implementations should assume `start()` is called only after `stop()`,
    and `stop()` must be safe to call when already stopped.
    """

    frame_width: int | None
    frame_height: int | None

    def start(self, rtsp_url: str) -> None: ...

    def stop(self) -> None: ...

    def read_frame(self, timeout_s: float) -> bytes | None: ...

    def is_running(self) -> bool: ...

    def exit_code(self) -> int | None: ...


class FfmpegFramePipeline:
    def __init__(
        self,
        *,
        output_dir: Path,
        frame_queue_size: int,
        rtsp_connect_timeout_s: float,
        rtsp_io_timeout_s: float,
        ffmpeg_flags: list[str],
        hwaccel_config: HardwareAccelConfig,
        hwaccel_failed: bool,
        on_frame: Callable[[], None],
        clock: Clock,
    ) -> None:
        self._output_dir = output_dir
        self._frame_queue_size = frame_queue_size
        self._rtsp_connect_timeout_s = rtsp_connect_timeout_s
        self._rtsp_io_timeout_s = rtsp_io_timeout_s
        self._ffmpeg_flags = ffmpeg_flags
        self._hwaccel_config = hwaccel_config
        self._hwaccel_failed = hwaccel_failed
        self._on_frame = on_frame
        self._clock = clock

        self._process: subprocess.Popen[bytes] | None = None
        self._stderr: Any | None = None
        self._reader_thread: Thread | None = None
        self._reader_stop: Event | None = None
        self._frame_queue: Queue[bytes] | None = None
        self.frame_width: int | None = None
        self.frame_height: int | None = None
        self._frame_size: int | None = None

    def start(self, rtsp_url: str) -> None:
        (
            self._process,
            self._stderr,
            self.frame_width,
            self.frame_height,
        ) = self._get_frame_pipe(rtsp_url)
        self._frame_size = int(self.frame_width) * int(self.frame_height)
        self._frame_queue = Queue(maxsize=self._frame_queue_size)
        self._reader_stop = Event()

        process = self._process
        frame_size = self._frame_size
        frame_queue = self._frame_queue
        stop_event = self._reader_stop

        def reader_loop() -> None:
            stdout = process.stdout if process else None
            if stdout is None:
                logger.error("Frame pipeline stdout is None")
                return

            while not stop_event.is_set():
                try:
                    raw = stdout.read(frame_size)
                except Exception:
                    logger.exception("Error reading from frame pipeline stdout")
                    return

                if not raw or len(raw) != frame_size:
                    return

                self._on_frame()

                try:
                    frame_queue.put_nowait(raw)
                except Full:
                    try:
                        _ = frame_queue.get_nowait()
                    except Empty:
                        pass
                    try:
                        frame_queue.put_nowait(raw)
                    except Full:
                        pass

        self._reader_thread = Thread(target=reader_loop, name="frame-reader", daemon=True)
        self._reader_thread.start()

    def stop(self) -> None:
        if self._reader_stop is not None:
            self._reader_stop.set()

        if self._process is not None:
            try:
                self._stop_process(self._process, "Frame pipeline", terminate_timeout_s=2)
            except Exception:
                logger.exception("Error stopping frame pipeline process")
            self._process = None

        if self._reader_thread is not None:
            try:
                self._reader_thread.join(timeout=5)
            except Exception:
                logger.exception("Error joining frame reader thread")
            self._reader_thread = None

        self._reader_stop = None

        if self._stderr is not None:
            try:
                self._stderr.close()
            except Exception:
                logger.exception("Error closing frame pipeline stderr log")
            self._stderr = None

        self._frame_queue = None
        self.frame_width = None
        self.frame_height = None
        self._frame_size = None

    def read_frame(self, timeout_s: float) -> bytes | None:
        if not self._frame_queue:
            return None
        try:
            return self._frame_queue.get(timeout=float(timeout_s))
        except Empty:
            return None

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def exit_code(self) -> int | None:
        if self._process is None:
            return None
        return self._process.poll()

    def _get_frame_pipe(self, rtsp_url: str) -> tuple[subprocess.Popen[bytes], Any, int, int]:
        detect_width, detect_height = 320, 240

        stderr_log = self._output_dir / "frame_pipeline.log"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        def _read_tail(path: Path, max_bytes: int = 4000) -> str:
            try:
                data = path.read_bytes()
            except Exception as exc:
                logger.warning("Failed to read stderr tail: %s", exc, exc_info=True)
                return ""
            if len(data) <= max_bytes:
                return data.decode(errors="replace")
            return data[-max_bytes:].decode(errors="replace")

        cmd = ["ffmpeg"]

        # 1. Global Flags (Hardware Acceleration)
        if self._hwaccel_config.is_available and not self._hwaccel_failed:
            hwaccel = self._hwaccel_config.hwaccel
            if hwaccel is not None:
                cmd.extend(["-hwaccel", hwaccel])
            if self._hwaccel_config.hwaccel_device:
                cmd.extend(["-hwaccel_device", self._hwaccel_config.hwaccel_device])
        elif self._hwaccel_failed:
            logger.info("Hardware acceleration disabled due to previous failures")

        # 2. Global Flags (Robustness & Logging)
        user_flags = self._ffmpeg_flags

        has_loglevel = any(x == "-loglevel" for x in user_flags)
        if not has_loglevel:
            cmd.extend(["-loglevel", "warning"])

        has_fflags = any(x == "-fflags" for x in user_flags)
        if not has_fflags:
            cmd.extend(["-fflags", "+genpts+igndts"])

        # Add all user flags to global scope.
        cmd.extend(user_flags)

        has_stimeout = any(x == "-stimeout" for x in user_flags)
        has_rw_timeout = any(x == "-rw_timeout" for x in user_flags)

        timeout_us_connect = str(int(max(0.1, self._rtsp_connect_timeout_s) * 1_000_000))
        timeout_us_io = str(int(max(0.1, self._rtsp_io_timeout_s) * 1_000_000))

        base_input_prefix = ["-rtsp_transport", "tcp"]
        base_input_args = [
            "-i",
            rtsp_url,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-vf",
            f"fps=10,scale={detect_width}:{detect_height}",
            "-an",
            "-",
        ]

        timeout_args: list[str] = []
        if not has_stimeout and self._rtsp_connect_timeout_s > 0:
            timeout_args.extend(["-stimeout", timeout_us_connect])
        if not has_rw_timeout and self._rtsp_io_timeout_s > 0:
            timeout_args.extend(["-rw_timeout", timeout_us_io])

        attempts: list[tuple[str, list[str]]] = []
        if timeout_args:
            attempts.append(("timeouts", base_input_prefix + timeout_args + base_input_args))
        attempts.append(("no_timeouts", base_input_prefix + base_input_args))

        process: subprocess.Popen[bytes] | None = None
        stderr_file: Any | None = None

        for label, extra_args in attempts:
            cmd_attempt = list(cmd) + extra_args
            logger.debug("Starting frame pipeline (%s), logging to: %s", label, stderr_log)
            safe_cmd = list(cmd_attempt)
            try:
                idx = safe_cmd.index("-i")
                safe_cmd[idx + 1] = _redact_rtsp_url(str(safe_cmd[idx + 1]))
            except Exception as exc:
                logger.warning(
                    "Failed to redact frame pipeline RTSP URL: %s",
                    exc,
                    exc_info=True,
                )
            logger.debug("Frame pipeline ffmpeg (%s): %s", label, _format_cmd(safe_cmd))

            try:
                stderr_file = open(stderr_log, "w")
                process = subprocess.Popen(
                    cmd_attempt,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                    bufsize=detect_width * detect_height,
                )
            except Exception:
                try:
                    if stderr_file:
                        stderr_file.close()
                except Exception as exc:
                    logger.warning("Failed to close stderr log: %s", exc, exc_info=True)
                logger.exception("Failed to start frame pipeline subprocess (%s)", label)
                continue

            self._clock.sleep(1)
            if process.poll() is None:
                return process, stderr_file, detect_width, detect_height

            try:
                stderr_file.close()
            except Exception as exc:
                logger.warning("Failed to close stderr log: %s", exc, exc_info=True)
            stderr_tail = _read_tail(stderr_log)
            timeout_option_error = (
                label == "timeouts" and bool(stderr_tail) and _is_timeout_option_error(stderr_tail)
            )
            if timeout_option_error:
                logger.warning(
                    "Frame pipeline died immediately (%s, exit code: %s); timeout options unsupported",
                    label,
                    process.returncode,
                )
                if stderr_tail:
                    logger.warning("Frame pipeline stderr tail (%s):\n%s", label, stderr_tail)
            else:
                logger.error(
                    "Frame pipeline died immediately (%s, exit code: %s)",
                    label,
                    process.returncode,
                )
                if stderr_tail:
                    logger.error("Frame pipeline stderr tail (%s):\n%s", label, stderr_tail)
            process = None
            stderr_file = None
            continue

        raise RuntimeError("Frame pipeline failed to start")

    def _stop_process(
        self, proc: subprocess.Popen[bytes], name: str, terminate_timeout_s: float
    ) -> None:
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=terminate_timeout_s)
        except subprocess.TimeoutExpired:
            logger.warning("%s did not terminate, killing (PID: %s)", name, proc.pid)
            proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:
                logger.exception("Failed waiting after kill for %s (PID: %s)", name, proc.pid)
