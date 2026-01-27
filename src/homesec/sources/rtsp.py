"""RTSP motion-detecting clip source.

Ported from motion_recorder.py to keep functionality within homesec.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any, Protocol, cast

import cv2
import numpy as np
import numpy.typing as npt

from homesec.models.clip import Clip
from homesec.models.source import RTSPSourceConfig
from homesec.sources.base import ThreadedClipSource

logger = logging.getLogger(__name__)


def _redact_rtsp_url(url: str) -> str:
    if "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if "@" not in rest:
        return url
    _creds, host = rest.split("@", 1)
    return f"{scheme}://***:***@{host}"


def _format_cmd(cmd: list[str]) -> str:
    try:
        return shlex.join([str(x) for x in cmd])
    except Exception as exc:
        logger.warning("Failed to format command with shlex.join: %s", exc, exc_info=True)
        return " ".join([str(x) for x in cmd])


def _is_timeout_option_error(stderr_text: str) -> bool:
    text = stderr_text.lower()
    return ("rw_timeout" in text and ("not found" in text or "unrecognized option" in text)) or (
        "stimeout" in text and ("not found" in text or "unrecognized option" in text)
    )


@dataclass
class HardwareAccelConfig:
    """Configuration for hardware-accelerated video decoding."""

    hwaccel: str | None
    hwaccel_device: str | None = None

    @property
    def is_available(self) -> bool:
        """Check if hardware acceleration is available."""
        return self.hwaccel is not None


class HardwareAccelDetector:
    """Detect available hardware acceleration options for ffmpeg."""

    @staticmethod
    def detect(rtsp_url: str) -> HardwareAccelConfig:
        """Detect the best available hardware acceleration method."""
        if Path("/dev/dri/renderD128").exists():
            if HardwareAccelDetector._test_hwaccel("vaapi"):
                config = HardwareAccelConfig(
                    hwaccel="vaapi",
                    hwaccel_device="/dev/dri/renderD128",
                )
                if HardwareAccelDetector._test_decode(rtsp_url, config):
                    return config
                logger.warning("VAAPI detected but failed to decode stream - disabling")

        if HardwareAccelDetector._check_nvidia():
            if HardwareAccelDetector._test_hwaccel("cuda"):
                config = HardwareAccelConfig(hwaccel="cuda")
                if HardwareAccelDetector._test_decode(rtsp_url, config):
                    return config
                logger.warning("CUDA detected but failed to decode stream - disabling")

        if platform.system() == "Darwin":
            if HardwareAccelDetector._test_hwaccel("videotoolbox"):
                config = HardwareAccelConfig(hwaccel="videotoolbox")
                if HardwareAccelDetector._test_decode(rtsp_url, config):
                    return config
                logger.warning("VideoToolbox detected but failed to decode stream - disabling")

        if HardwareAccelDetector._test_hwaccel("qsv"):
            config = HardwareAccelConfig(hwaccel="qsv")
            if HardwareAccelDetector._test_decode(rtsp_url, config):
                return config
            logger.warning("QSV detected but failed to decode stream - disabling")

        logger.info("Using software decoding (no working hardware acceleration found)")
        return HardwareAccelConfig(hwaccel=None)

    @staticmethod
    def _test_hwaccel(method: str) -> bool:
        """Test if a hardware acceleration method is available in ffmpeg."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-hwaccels"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            return method in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False

    @staticmethod
    def _test_decode(rtsp_url: str, config: HardwareAccelConfig) -> bool:
        """Test if hardware acceleration works by decoding a few frames."""
        cmd = ["ffmpeg"]

        if config.hwaccel:
            cmd.extend(["-hwaccel", config.hwaccel])
            if config.hwaccel_device:
                cmd.extend(["-hwaccel_device", config.hwaccel_device])

        cmd.extend(
            [
                "-rtsp_transport",
                "tcp",
                "-i",
                rtsp_url,
                "-frames:v",
                "5",
                "-f",
                "null",
                "-",
            ]
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return True
            if any(
                err in result.stderr
                for err in (
                    "No VA display found",
                    "Device creation failed",
                    "No device available for decoder",
                    "Failed to initialise VAAPI",
                    "Cannot load",
                )
            ):
                return False
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return True
        except Exception as exc:
            logger.warning("VAAPI check failed: %s", exc, exc_info=True)
            return False

    @staticmethod
    def _check_nvidia() -> bool:
        """Check if NVIDIA GPU is available."""
        try:
            subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                check=True,
                timeout=2,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False


class Clock(Protocol):
    def now(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class MotionDetector:
    def __init__(
        self,
        *,
        pixel_threshold: int,
        min_changed_pct: float,
        blur_kernel: int,
        debug: bool,
    ) -> None:
        self._pixel_threshold = int(pixel_threshold)
        self._min_changed_pct = float(min_changed_pct)
        self._blur_kernel = int(blur_kernel)
        self._debug = bool(debug)

        self._prev_frame: npt.NDArray[np.uint8] | None = None
        self._last_changed_pct = 0.0
        self._last_changed_pixels = 0
        self._debug_frame_count = 0

    @property
    def last_changed_pct(self) -> float:
        return self._last_changed_pct

    @property
    def last_changed_pixels(self) -> int:
        return self._last_changed_pixels

    def reset(self) -> None:
        self._prev_frame = None
        self._last_changed_pct = 0.0
        self._last_changed_pixels = 0
        self._debug_frame_count = 0

    def detect(self, frame: npt.NDArray[np.uint8], *, threshold: float | None = None) -> bool:
        if frame.ndim == 3:
            gray = cast(npt.NDArray[np.uint8], cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            gray = frame

        if self._blur_kernel > 1:
            gray = cast(
                npt.NDArray[np.uint8],
                cv2.GaussianBlur(gray, (self._blur_kernel, self._blur_kernel), 0),
            )

        if self._prev_frame is None:
            self._prev_frame = gray
            self._last_changed_pct = 0.0
            self._last_changed_pixels = 0
            return False

        diff = cv2.absdiff(self._prev_frame, gray)
        _, mask = cv2.threshold(diff, self._pixel_threshold, 255, cv2.THRESH_BINARY)
        changed_pixels = int(cv2.countNonZero(mask))

        total_pixels = int(gray.shape[0]) * int(gray.shape[1])
        changed_pct = (changed_pixels / total_pixels * 100.0) if total_pixels else 0.0

        self._prev_frame = gray
        self._last_changed_pct = changed_pct
        self._last_changed_pixels = changed_pixels

        if threshold is None:
            threshold = self._min_changed_pct
        if threshold < 0:
            threshold = 0.0

        motion = changed_pct >= float(threshold)

        if self._debug:
            self._debug_frame_count += 1
            if self._debug_frame_count % 100 == 0:
                logger.debug(
                    "Motion check: changed_pct=%.3f%% changed_px=%s pixel_threshold=%s min_changed_pct=%.3f%% blur=%s",
                    changed_pct,
                    changed_pixels,
                    self._pixel_threshold,
                    self._min_changed_pct,
                    self._blur_kernel,
                )

        return motion


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
            logger.error(
                "Frame pipeline died immediately (%s, exit code: %s)", label, process.returncode
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
    ) -> None:
        self._rtsp_url = rtsp_url
        self._ffmpeg_flags = ffmpeg_flags
        self._rtsp_connect_timeout_s = rtsp_connect_timeout_s
        self._rtsp_io_timeout_s = rtsp_io_timeout_s
        self._clock = clock

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
        has_stimeout = any(x == "-stimeout" for x in user_flags)
        has_rw_timeout = any(x == "-rw_timeout" for x in user_flags)
        timeout_us_connect = str(int(max(0.1, self._rtsp_connect_timeout_s) * 1_000_000))
        timeout_us_io = str(int(max(0.1, self._rtsp_io_timeout_s) * 1_000_000))

        timeout_args: list[str] = []
        if not has_stimeout and self._rtsp_connect_timeout_s > 0:
            timeout_args.extend(["-stimeout", timeout_us_connect])
        if not has_rw_timeout and self._rtsp_io_timeout_s > 0:
            timeout_args.extend(["-rw_timeout", timeout_us_io])

        cmd_tail = ["-i", self._rtsp_url, "-c", "copy", "-f", "mp4", "-y"]

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

        has_fps_mode = any(x == "-fps_mode" or x == "-vsync" for x in user_flags)
        if not has_fps_mode:
            cmd_tail.extend(["-vsync", "0"])

        # Add user flags last so they can potentially override or add to the above
        cmd_tail.extend(user_flags)
        cmd_tail.extend([str(output_file)])

        attempts: list[tuple[str, list[str]]] = []
        if timeout_args:
            attempts.append(("timeouts", timeout_args))
        attempts.append(("no_timeouts" if timeout_args else "default", []))

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
                    return proc

                logger.error(
                    "Recording process died immediately (%s, exit code: %s)",
                    label,
                    proc.returncode,
                )
                logger.error("Check logs at: %s", stderr_log)
                stderr_tail = _read_tail(stderr_log)
                if stderr_tail:
                    redacted_tail = stderr_tail.replace(
                        self._rtsp_url, _redact_rtsp_url(self._rtsp_url)
                    )
                    logger.error("Recording stderr tail (%s):\n%s", label, redacted_tail)
                    if label == "timeouts" and _is_timeout_option_error(stderr_tail):
                        logger.warning(
                            "Recording ffmpeg missing timeout options; retrying without timeouts"
                        )
                        continue
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


class RTSPRunState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    STALLED = "stalled"
    RECONNECTING = "reconnecting"


class RTSPSource(ThreadedClipSource):
    """RTSP clip source with motion detection.

    Uses ffmpeg for frame extraction and recording; detects motion from
    downscaled grayscale frames and emits clips when recordings finish.
    """

    def __init__(
        self,
        config: RTSPSourceConfig,
        camera_name: str,
        *,
        frame_pipeline: FramePipeline | None = None,
        recorder: Recorder | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Initialize RTSP source."""
        super().__init__()
        self._clock = clock or SystemClock()
        rtsp_url = config.rtsp_url
        if config.rtsp_url_env:
            env_value = os.getenv(config.rtsp_url_env)
            if env_value:
                rtsp_url = env_value
        if not rtsp_url:
            raise ValueError("rtsp_url_env or rtsp_url required for RTSP source")

        self.rtsp_url = rtsp_url

        detect_rtsp_url = config.detect_rtsp_url
        if config.detect_rtsp_url_env:
            env_value = os.getenv(config.detect_rtsp_url_env)
            if env_value:
                detect_rtsp_url = env_value

        derived_detect = self._derive_detect_rtsp_url(self.rtsp_url)
        if detect_rtsp_url:
            self.detect_rtsp_url = detect_rtsp_url
            self._detect_rtsp_url_source = "explicit"
        elif derived_detect:
            self.detect_rtsp_url = derived_detect
            self._detect_rtsp_url_source = "derived_subtype=1"
        else:
            self.detect_rtsp_url = self.rtsp_url
            self._detect_rtsp_url_source = "same_as_rtsp_url"

        self.output_dir = Path(config.output_dir)
        sanitized_name = self._sanitize_camera_name(camera_name)
        self.camera_name = sanitized_name or camera_name

        self.pixel_threshold = int(config.pixel_threshold)
        self.min_changed_pct = float(config.min_changed_pct)
        self.recording_sensitivity_factor = float(config.recording_sensitivity_factor)
        self.blur_kernel = self._normalize_blur_kernel(config.blur_kernel)
        self.motion_stop_delay = float(config.stop_delay)
        self.max_recording_s = float(config.max_recording_s)
        self.max_reconnect_attempts = int(config.max_reconnect_attempts)
        self.detect_fallback_attempts = int(config.detect_fallback_attempts)
        self.debug_motion = bool(config.debug_motion)
        self.heartbeat_s = float(config.heartbeat_s)
        self.frame_timeout_s = float(config.frame_timeout_s)
        self.frame_queue_size = int(config.frame_queue_size)
        self.reconnect_backoff_s = float(config.reconnect_backoff_s)
        self.rtsp_connect_timeout_s = float(config.rtsp_connect_timeout_s)
        self.rtsp_io_timeout_s = float(config.rtsp_io_timeout_s)
        self.ffmpeg_flags = list(config.ffmpeg_flags)

        if config.disable_hwaccel:
            logger.info("Hardware acceleration manually disabled")
            self.hwaccel_config = HardwareAccelConfig(hwaccel=None)
            self._hwaccel_failed = True
        else:
            logger.info("Testing hardware acceleration with camera stream...")
            self.hwaccel_config = HardwareAccelDetector.detect(self.detect_rtsp_url)
            self._hwaccel_failed = not self.hwaccel_config.is_available

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.recording_process: subprocess.Popen[bytes] | None = None
        self.last_motion_time: float | None = None
        self.recording_start_time: float | None = None
        self.recording_start_wall: datetime | None = None
        self.output_file: Path | None = None
        self._stderr_log: Path | None = None
        self._recording_id: str | None = None
        self._recording_restart_backoff_s = 0.0
        self._recording_restart_backoff_max_s = 10.0
        self._recording_restart_base_s = 0.5
        self._recording_next_restart_at: float | None = None
        self._stall_grace_until: float | None = None
        self._motion_detector = MotionDetector(
            pixel_threshold=self.pixel_threshold,
            min_changed_pct=self.min_changed_pct,
            blur_kernel=self.blur_kernel,
            debug=self.debug_motion,
        )

        self._frame_pipeline: FramePipeline = frame_pipeline or FfmpegFramePipeline(
            output_dir=self.output_dir,
            frame_queue_size=self.frame_queue_size,
            rtsp_connect_timeout_s=self.rtsp_connect_timeout_s,
            rtsp_io_timeout_s=self.rtsp_io_timeout_s,
            ffmpeg_flags=self.ffmpeg_flags,
            hwaccel_config=self.hwaccel_config,
            hwaccel_failed=self._hwaccel_failed,
            on_frame=self._touch_heartbeat,
            clock=self._clock,
        )
        self._recorder: Recorder = recorder or FfmpegRecorder(
            rtsp_url=self.rtsp_url,
            ffmpeg_flags=self.ffmpeg_flags,
            rtsp_connect_timeout_s=self.rtsp_connect_timeout_s,
            rtsp_io_timeout_s=self.rtsp_io_timeout_s,
            clock=self._clock,
        )
        self._run_state = RTSPRunState.IDLE
        self._motion_rtsp_url = self.detect_rtsp_url
        self._detect_stream_available = self.detect_rtsp_url != self.rtsp_url
        self._detect_fallback_active = False
        self._detect_fallback_deferred = False
        self._detect_failure_count = 0
        self._detect_next_probe_at: float | None = None
        self._detect_probe_interval_s = 25.0
        self._detect_probe_backoff_s = self._detect_probe_interval_s
        self._detect_probe_backoff_max_s = 60.0
        self.reconnect_count = 0
        self.last_successful_frame = self._clock.now()

        logger.info(
            "RTSPSource initialized: camera=%s, output_dir=%s",
            self.camera_name,
            self.output_dir,
        )

    def is_healthy(self) -> bool:
        """Check if source is healthy."""
        if not self._thread_is_healthy():
            return False

        age = self._clock.now() - self.last_successful_frame
        return age < (self.frame_timeout_s * 3)

    def _touch_heartbeat(self) -> None:
        self.last_successful_frame = self._clock.now()
        super()._touch_heartbeat()

    def _stop_timeout(self) -> float:
        return 10.0

    def _on_start(self) -> None:
        logger.info("Starting RTSPSource: %s", self.camera_name)

    def _on_stop(self) -> None:
        logger.info("Stopping RTSPSource...")

    def _on_stopped(self) -> None:
        logger.info("RTSPSource stopped")

    def _derive_detect_rtsp_url(self, rtsp_url: str) -> str | None:
        if "subtype=0" in rtsp_url:
            return rtsp_url.replace("subtype=0", "subtype=1")
        return None

    def _sanitize_camera_name(self, name: str | None) -> str | None:
        if not name:
            return None
        raw = str(name).strip()
        if not raw:
            return None
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
        return cleaned or None

    def _recording_prefix(self) -> str:
        if not self.camera_name:
            return ""
        return f"{self.camera_name}_"

    def _make_recording_paths(self, timestamp: str) -> tuple[Path, Path]:
        prefix = self._recording_prefix()
        output_file = self.output_dir / f"{prefix}motion_{timestamp}.mp4"
        stderr_log = self.output_dir / f"{prefix}recording_{timestamp}.log"
        return output_file, stderr_log

    def _telemetry_common_fields(self) -> dict[str, object]:
        return {
            "pixel_threshold": self.pixel_threshold,
            "min_changed_pct": self.min_changed_pct,
            "recording_sensitivity_factor": self.recording_sensitivity_factor,
            "blur_kernel": self.blur_kernel,
            "stop_delay": self.motion_stop_delay,
            "max_recording_s": self.max_recording_s,
            "detect_fallback_attempts": self.detect_fallback_attempts,
            "hwaccel": self.hwaccel_config.hwaccel,
            "hwaccel_device": self.hwaccel_config.hwaccel_device,
        }

    def _event_extra(self, event_type: str, **fields: object) -> dict[str, object]:
        extra: dict[str, object] = {
            "kind": "event",
            "event_type": event_type,
            "camera_name": self.camera_name,
            "recording_id": self._recording_id,
        }
        extra.update(self._telemetry_common_fields())
        extra.update(fields)
        return extra

    def _probe_stream_info(
        self,
        rtsp_url: str,
        *,
        timeout_s: float = 10.0,
    ) -> dict[str, object] | None:
        base_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate",
            "-of",
            "json",
            "-rtsp_transport",
            "tcp",
        ]
        timeout_args: list[str] = []
        if self.rtsp_connect_timeout_s > 0:
            timeout_us_connect = str(int(max(0.1, self.rtsp_connect_timeout_s) * 1_000_000))
            timeout_args.extend(["-stimeout", timeout_us_connect])
        if self.rtsp_io_timeout_s > 0:
            timeout_us_io = str(int(max(0.1, self.rtsp_io_timeout_s) * 1_000_000))
            timeout_args.extend(["-rw_timeout", timeout_us_io])

        attempts: list[tuple[str, list[str]]] = []
        if timeout_args:
            attempts.append(("timeouts", base_cmd + timeout_args + [rtsp_url]))
        attempts.append(("no_timeouts" if timeout_args else "default", base_cmd + [rtsp_url]))

        for label, cmd in attempts:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    check=False,
                )
                if result.returncode != 0:
                    if _is_timeout_option_error(result.stderr):
                        logger.debug("ffprobe missing timeout options (%s), retrying", label)
                        continue
                    logger.debug("ffprobe failed (%s) with exit code %s", label, result.returncode)
                    break

                data = json.loads(result.stdout)
                if not data.get("streams"):
                    continue
                stream = data["streams"][0]
                return {
                    "codec_name": stream.get("codec_name"),
                    "width": stream.get("width"),
                    "height": stream.get("height"),
                    "avg_frame_rate": stream.get("avg_frame_rate"),
                }
            except Exception as exc:
                logger.warning("Failed to probe stream info: %s", exc, exc_info=True)
                return None

        return None

    def _redact_rtsp_url(self, url: str) -> str:
        return _redact_rtsp_url(url)

    def _format_cmd(self, cmd: list[str]) -> str:
        return _format_cmd(cmd)

    def detect_motion(
        self, frame: npt.NDArray[np.uint8], *, threshold: float | None = None
    ) -> bool:
        """Return True if motion detected in frame."""
        return self._motion_detector.detect(frame, threshold=threshold)

    def check_recording_health(self) -> bool:
        """Check if recording process is still alive."""
        if self.recording_process and not self._recorder.is_alive(self.recording_process):
            proc = self.recording_process
            output_file = self.output_file
            exit_code = getattr(proc, "returncode", None)
            logger.warning("Recording process died unexpectedly (exit code: %s)", exit_code)
            if output_file:
                logger.error(
                    "Recording process died",
                    extra=self._event_extra(
                        "recording_process_died",
                        recording_id=output_file.name,
                        recording_path=str(output_file),
                        exit_code=exit_code,
                        pid=proc.pid,
                    ),
                )

            log_file: Path | None = None
            if self._stderr_log and self._stderr_log.exists():
                log_file = self._stderr_log
            elif output_file:
                stem = output_file.stem
                if "motion_" in stem:
                    prefix_part, timestamp = stem.split("motion_", 1)
                    candidate = output_file.parent / f"{prefix_part}recording_{timestamp}.log"
                    if candidate.exists():
                        log_file = candidate

            if log_file:
                try:
                    with open(log_file) as f:
                        error_lines = f.read()
                        if error_lines:
                            logger.warning("Recording error log:\n%s", error_lines[-1000:])
                except Exception as e:
                    logger.warning("Could not read log file: %s", e, exc_info=True)

            start_wall = self.recording_start_wall
            start_mono = self.recording_start_time
            self.recording_process = None
            self.output_file = None
            self._stderr_log = None
            self.recording_start_time = None
            self.recording_start_wall = None
            self._recording_id = None

            self._stop_recording_process(proc, output_file)
            self._finalize_clip(output_file, start_wall, start_mono)
            return False
        return True

    def start_recording(self) -> None:
        """Start ffmpeg recording process with audio."""
        if self.recording_process:
            return
        now = self._clock.now()
        if not self._recording_backoff_ready(now):
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file, stderr_log = self._make_recording_paths(timestamp)

        proc = self._recorder.start(output_file, stderr_log)
        if not proc:
            self._bump_recording_backoff(now)
            logger.error(
                "Recording failed to start",
                extra=self._event_extra(
                    "recording_start_error",
                    recording_id=output_file.name,
                    recording_path=str(output_file),
                    stderr_log=str(stderr_log),
                ),
            )
            return

        self.recording_process = proc
        self.output_file = output_file
        self._stderr_log = stderr_log
        self.recording_start_time = self._clock.now()
        self.recording_start_wall = datetime.now()
        self._recording_id = output_file.name
        self._reset_recording_backoff()

        logger.info("Started recording: %s (PID: %s)", output_file, proc.pid)
        logger.debug("Recording logs: %s", stderr_log)
        logger.info(
            "Recording started",
            extra=self._event_extra(
                "recording_start",
                recording_id=output_file.name,
                recording_path=str(output_file),
                stderr_log=str(stderr_log),
                pid=proc.pid,
                detect_stream_source=getattr(self, "_detect_rtsp_url_source", None),
                detect_stream_is_same=(self.detect_rtsp_url == self.rtsp_url),
            ),
        )

    def stop_recording(self) -> None:
        """Stop ffmpeg recording process."""
        if not self.recording_process:
            return

        proc = self.recording_process
        output_file = self.output_file
        started_at = self.recording_start_time
        started_wall = self.recording_start_wall
        self.recording_process = None
        self.output_file = None
        self._stderr_log = None
        self.recording_start_time = None
        self.recording_start_wall = None
        self._recording_id = None

        self._stop_recording_process(proc, output_file)
        self._finalize_clip(output_file, started_wall, started_at)

        if output_file:
            duration_s = (self._clock.now() - started_at) if started_at else None
            logger.info(
                "Recording stopped",
                extra=self._event_extra(
                    "recording_stop",
                    recording_id=output_file.name,
                    recording_path=str(output_file),
                    duration_s=duration_s,
                    last_changed_pct=self._motion_detector.last_changed_pct,
                    last_changed_pixels=self._motion_detector.last_changed_pixels,
                ),
            )

    def _stop_recording_process(
        self, proc: subprocess.Popen[bytes], output_file: Path | None
    ) -> None:
        try:
            self._recorder.stop(proc, output_file)
        except Exception:
            logger.exception("Failed while stopping recording process (PID: %s)", proc.pid)

    def _rotate_recording_if_needed(self) -> None:
        if self.recording_process is None or self.recording_start_time is None:
            return

        now = self._clock.now()
        if (now - self.recording_start_time) < self.max_recording_s:
            return

        if self.last_motion_time is None:
            return

        if (now - self.last_motion_time) > self.motion_stop_delay:
            return

        old_proc = self.recording_process
        old_output = self.output_file
        old_started_wall = self.recording_start_wall
        old_started_mono = self.recording_start_time

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_output, new_log = self._make_recording_paths(timestamp)

        logger.info(
            "Max recording length reached (%.1fs), rotating to: %s",
            self.max_recording_s,
            new_output,
        )

        new_proc = self._recorder.start(new_output, new_log)
        if new_proc:
            self.last_motion_time = now
            self.recording_process = new_proc
            self.output_file = new_output
            self._stderr_log = new_log
            self.recording_start_time = now
            self.recording_start_wall = datetime.now()
            self._recording_id = new_output.name

            if old_output:
                logger.info(
                    "Recording rotated",
                    extra=self._event_extra(
                        "recording_rotate",
                        recording_id=old_output.name,
                        recording_path=str(old_output),
                        new_recording_id=new_output.name,
                        new_recording_path=str(new_output),
                    ),
                )

            self._stop_recording_process(old_proc, old_output)
            self._finalize_clip(old_output, old_started_wall, old_started_mono)
            return

        logger.warning("Rotation start failed; stopping current recording and retrying")
        self.recording_process = None
        self.output_file = None
        self.recording_start_time = None
        self.recording_start_wall = None
        self._recording_id = None
        self._stop_recording_process(old_proc, old_output)
        self._finalize_clip(old_output, old_started_wall, old_started_mono)
        self.start_recording()

    def _start_frame_pipeline(self) -> None:
        try:
            self._frame_pipeline.stop()
        except Exception:
            logger.exception("Error stopping frame pipeline before start")
        self._frame_pipeline.start(self._motion_rtsp_url)
        self._motion_detector.reset()

    def _stop_frame_pipeline(self) -> None:
        self._frame_pipeline.stop()
        self._motion_detector.reset()

    def _wait_for_first_frame(self, timeout_s: float) -> bool:
        return self._frame_pipeline.read_frame(timeout_s) is not None

    def _reconnect_frame_pipeline(self, *, aggressive: bool) -> bool:
        initial_backoff_s = 0.2 if aggressive else float(self.reconnect_backoff_s)
        backoff_s = initial_backoff_s
        backoff_cap_s = 10.0 if aggressive else float(self.reconnect_backoff_s)

        first_attempt = True
        while not self._stop_event.is_set():
            self.reconnect_count += 1

            if self.max_reconnect_attempts == 0:
                logger.warning(
                    "Reconnect attempt %s (mode=%s max=inf)...",
                    self.reconnect_count,
                    "aggressive" if aggressive else "normal",
                    extra=self._event_extra(
                        "rtsp_reconnect_attempt",
                        attempt=self.reconnect_count,
                        max_attempts=None,
                        mode="aggressive" if aggressive else "normal",
                        detect_is_fallback=self._detect_fallback_active,
                        motion_rtsp_url=self._redact_rtsp_url(self._motion_rtsp_url),
                    ),
                )
            else:
                logger.warning(
                    "Reconnect attempt %s/%s (mode=%s)...",
                    self.reconnect_count,
                    self.max_reconnect_attempts,
                    "aggressive" if aggressive else "normal",
                    extra=self._event_extra(
                        "rtsp_reconnect_attempt",
                        attempt=self.reconnect_count,
                        max_attempts=self.max_reconnect_attempts,
                        mode="aggressive" if aggressive else "normal",
                        detect_is_fallback=self._detect_fallback_active,
                        motion_rtsp_url=self._redact_rtsp_url(self._motion_rtsp_url),
                    ),
                )

            sleep_s = self._reconnect_sleep_s(
                backoff_s=backoff_s,
                aggressive=aggressive,
                first_attempt=first_attempt,
            )
            first_attempt = False

            if sleep_s > 0:
                logger.debug("Waiting %.2fs before reconnect...", sleep_s)
                self._clock.sleep(sleep_s)

            try:
                self._start_frame_pipeline()
            except Exception as exc:
                logger.warning("Failed to restart frame pipeline: %s", exc, exc_info=True)
                triggered_fallback = self._note_detect_failure(self._clock.now())
                if self._reconnect_exhausted(aggressive=aggressive):
                    return False
                if triggered_fallback:
                    backoff_s = initial_backoff_s
                    first_attempt = True
                    continue
                backoff_s = self._bump_reconnect_backoff(
                    backoff_s,
                    backoff_cap_s,
                    aggressive=aggressive,
                )
                continue

            startup_timeout_s = min(2.0, max(0.5, float(self.frame_timeout_s)))
            if self._wait_for_first_frame(startup_timeout_s):
                logger.info(
                    "Reconnected successfully",
                    extra=self._event_extra(
                        "rtsp_reconnect_success",
                        attempt=self.reconnect_count,
                        mode="aggressive" if aggressive else "normal",
                        detect_is_fallback=self._detect_fallback_active,
                        motion_rtsp_url=self._redact_rtsp_url(self._motion_rtsp_url),
                    ),
                )
                self.reconnect_count = 0
                self._detect_failure_count = 0
                self._detect_fallback_deferred = False
                return True

            logger.warning("Frame pipeline restarted but still no frames; retrying...")
            self._note_detect_failure(self._clock.now())
            try:
                self._stop_frame_pipeline()
            except Exception as exc:
                logger.warning(
                    "Failed to stop frame pipeline after reconnect: %s",
                    exc,
                    exc_info=True,
                )
            if self._reconnect_exhausted(aggressive=aggressive):
                return False
            backoff_s = self._bump_reconnect_backoff(
                backoff_s,
                backoff_cap_s,
                aggressive=aggressive,
            )

        return False

    def _reconnect_sleep_s(
        self,
        *,
        backoff_s: float,
        aggressive: bool,
        first_attempt: bool,
    ) -> float:
        if first_attempt and aggressive:
            return 0.0
        jitter = random.uniform(0.0, min(0.25, backoff_s * 0.25))
        return backoff_s + jitter

    def _bump_reconnect_backoff(
        self,
        backoff_s: float,
        backoff_cap_s: float,
        *,
        aggressive: bool,
    ) -> float:
        if not aggressive:
            return backoff_s
        return min(backoff_s * 1.6, backoff_cap_s)

    def _recording_backoff_ready(self, now: float) -> bool:
        next_retry = self._recording_next_restart_at
        if next_retry is None or now >= next_retry:
            return True
        remaining = next_retry - now
        logger.debug("Recording restart backoff active (%.2fs remaining)", remaining)
        return False

    def _bump_recording_backoff(self, now: float) -> None:
        if self._recording_restart_backoff_s <= 0:
            self._recording_restart_backoff_s = self._recording_restart_base_s
        else:
            self._recording_restart_backoff_s = min(
                self._recording_restart_backoff_s * 1.6,
                self._recording_restart_backoff_max_s,
            )
        self._recording_next_restart_at = now + self._recording_restart_backoff_s

    def _reset_recording_backoff(self) -> None:
        self._recording_restart_backoff_s = 0.0
        self._recording_next_restart_at = None

    def _reconnect_exhausted(self, *, aggressive: bool) -> bool:
        if self.max_reconnect_attempts <= 0:
            return False
        if self.reconnect_count < self.max_reconnect_attempts:
            return False
        logger.error(
            "Max reconnect attempts reached. Camera may be offline.",
            extra=self._event_extra(
                "rtsp_reconnect_failed",
                attempt=self.reconnect_count,
                max_attempts=self.max_reconnect_attempts,
                mode="aggressive" if aggressive else "normal",
                detect_is_fallback=self._detect_fallback_active,
                motion_rtsp_url=self._redact_rtsp_url(self._motion_rtsp_url),
            ),
        )
        return True

    def _note_detect_failure(self, now: float) -> bool:
        if not self._detect_stream_available:
            return False
        if self._detect_fallback_active:
            return False
        if self.detect_fallback_attempts <= 0:
            return False

        self._detect_failure_count = min(
            self._detect_failure_count + 1,
            self.detect_fallback_attempts,
        )
        if self._detect_failure_count < self.detect_fallback_attempts:
            return False

        if self.recording_process is not None:
            if not self._detect_fallback_deferred:
                logger.warning(
                    "Detect fallback deferred while recording",
                    extra=self._event_extra(
                        "detect_fallback_deferred",
                        detect_stream_source=getattr(self, "_detect_rtsp_url_source", None),
                        detect_stream_is_same=False,
                    ),
                )
                self._detect_fallback_deferred = True
            return False

        self._activate_detect_fallback(now)
        return True

    def _activate_detect_fallback(self, now: float) -> None:
        self._detect_fallback_active = True
        self._detect_fallback_deferred = False
        self._detect_failure_count = 0
        self._motion_rtsp_url = self.rtsp_url
        self._detect_next_probe_at = now + self._detect_probe_interval_s
        self._detect_probe_backoff_s = self._detect_probe_interval_s
        logger.warning(
            "Detect stream failed; falling back to main RTSP stream",
            extra=self._event_extra(
                "detect_fallback_enabled",
                detect_stream_source=getattr(self, "_detect_rtsp_url_source", None),
                detect_stream_is_same=False,
            ),
        )

    def _schedule_detect_probe(self, now: float) -> None:
        self._detect_next_probe_at = now + self._detect_probe_backoff_s
        self._detect_probe_backoff_s = min(
            self._detect_probe_backoff_s * 1.6,
            self._detect_probe_backoff_max_s,
        )

    def _maybe_recover_detect_stream(self, now: float) -> bool:
        if not self._detect_fallback_active:
            return False
        if not self._detect_stream_available:
            return False
        if self._detect_next_probe_at is not None and now < self._detect_next_probe_at:
            return False

        info = self._probe_stream_info(self.detect_rtsp_url, timeout_s=3.0)
        if not info or not info.get("width") or not info.get("height"):
            self._schedule_detect_probe(now)
            return False

        logger.info(
            "Detect stream recovered; attempting switch back",
            extra=self._event_extra(
                "detect_fallback_recovering",
                detect_stream_source=getattr(self, "_detect_rtsp_url_source", None),
                detect_stream_is_same=False,
            ),
        )

        self._motion_rtsp_url = self.detect_rtsp_url
        try:
            self._start_frame_pipeline()
            startup_timeout_s = min(2.0, max(0.5, float(self.frame_timeout_s)))
            if self._wait_for_first_frame(startup_timeout_s):
                self._detect_fallback_active = False
                self._detect_next_probe_at = None
                self._detect_probe_backoff_s = self._detect_probe_interval_s
                logger.info(
                    "Detect stream restored; using detect stream again",
                    extra=self._event_extra(
                        "detect_fallback_recovered",
                        detect_stream_source=getattr(self, "_detect_rtsp_url_source", None),
                        detect_stream_is_same=False,
                    ),
                )
                return True
        except Exception as exc:
            logger.warning("Detect stream switch failed: %s", exc, exc_info=True)

        self._motion_rtsp_url = self.rtsp_url
        self._detect_fallback_active = True
        self._schedule_detect_probe(now)
        try:
            self._start_frame_pipeline()
        except Exception as exc:
            logger.warning(
                "Failed to restart fallback pipeline after detect switch: %s",
                exc,
                exc_info=True,
            )
        return False

    def _keepalive_threshold(self) -> float:
        if self.recording_sensitivity_factor <= 0:
            return self.min_changed_pct
        return max(0.0, self.min_changed_pct / self.recording_sensitivity_factor)

    def _set_run_state(self, state: RTSPRunState) -> None:
        if self._run_state == state:
            return
        logger.debug("RTSP state change: %s -> %s", self._run_state, state)
        self._run_state = state

    def _ensure_recording(self, now: float) -> None:
        if self.recording_process:
            if not self.check_recording_health():
                logger.warning(
                    "Recording died, attempting restart",
                    extra=self._event_extra(
                        "recording_restart_attempt",
                        reason="health_check",
                    ),
                )
                if self.last_motion_time is not None and (
                    (now - self.last_motion_time) <= self.motion_stop_delay
                ):
                    self.start_recording()
                    if self.recording_process:
                        logger.info(
                            "Recording restarted",
                            extra=self._event_extra(
                                "recording_restart",
                                reason="health_check",
                            ),
                        )
                else:
                    self.last_motion_time = None
            return

        if (
            self.last_motion_time is not None
            and (now - self.last_motion_time) <= self.motion_stop_delay
        ):
            logger.warning(
                "Recording missing while motion is recent; restarting",
                extra=self._event_extra(
                    "recording_restart_attempt",
                    reason="missing",
                ),
            )
            self.start_recording()
            if self.recording_process:
                logger.info(
                    "Recording restarted",
                    extra=self._event_extra(
                        "recording_restart",
                        reason="missing",
                    ),
                )
        elif (
            self.last_motion_time is not None
            and (now - self.last_motion_time) > self.motion_stop_delay
        ):
            self.last_motion_time = None

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up...")
        try:
            self.stop_recording()
        except Exception:
            logger.exception("Error stopping recording")
        try:
            self._stop_frame_pipeline()
        except Exception:
            logger.exception("Error stopping frame pipeline")

    def _finalize_clip(
        self,
        output_file: Path | None,
        started_wall: datetime | None,
        started_mono: float | None,
    ) -> None:
        if output_file is None:
            return
        try:
            if not output_file.exists() or output_file.stat().st_size == 0:
                return
        except Exception as exc:
            logger.warning(
                "Failed to stat output clip %s: %s",
                output_file,
                exc,
                exc_info=True,
            )
            return

        end_ts = datetime.now()
        if started_wall is None:
            if started_mono is not None:
                duration_s = time.monotonic() - started_mono
                start_ts = end_ts - timedelta(seconds=duration_s)
            else:
                start_ts = end_ts
                duration_s = 0.0
        else:
            start_ts = started_wall
            duration_s = (end_ts - start_ts).total_seconds()

        clip = Clip(
            clip_id=output_file.stem,
            camera_name=self.camera_name,
            local_path=output_file,
            start_ts=start_ts,
            end_ts=end_ts,
            duration_s=duration_s,
            source_type="rtsp",
        )

        self._emit_clip(clip)

    @staticmethod
    def _normalize_blur_kernel(blur_kernel: int) -> int:
        kernel = int(blur_kernel)
        if kernel < 0:
            return 0
        if kernel % 2 == 0 and kernel != 0:
            return kernel + 1
        return kernel

    def _log_startup_info(self) -> None:
        logger.info("Connecting to camera...")
        logger.info(
            "Motion: pixel_threshold=%s min_changed_pct=%.3f%% blur=%s stop_delay=%.1fs max_recording_s=%.1fs",
            self.pixel_threshold,
            self.min_changed_pct,
            self.blur_kernel,
            self.motion_stop_delay,
            self.max_recording_s,
        )
        logger.info(
            "Motion thresholds: idle=%.3f%% recording=%.3f%% (recording_sensitivity_factor=%.2f)",
            self.min_changed_pct,
            self._keepalive_threshold(),
            self.recording_sensitivity_factor,
        )
        logger.info("Max reconnect attempts: %s", self.max_reconnect_attempts)
        if self.max_reconnect_attempts == 0:
            logger.info("Reconnect policy: retry forever")
        if self._detect_stream_available:
            logger.info("Detect fallback attempts: %s", self.detect_fallback_attempts)

        if self.hwaccel_config.is_available:
            device_info = (
                f" (device: {self.hwaccel_config.hwaccel_device})"
                if self.hwaccel_config.hwaccel_device
                else ""
            )
            logger.info("Hardware acceleration: %s%s", self.hwaccel_config.hwaccel, device_info)
        else:
            logger.info("Hardware acceleration: disabled (using software decoding)")

        logger.info("Detecting camera resolution...")
        info = self._probe_stream_info(self.rtsp_url)
        width = info.get("width") if info else None
        height = info.get("height") if info else None
        if isinstance(width, (int, str)) and isinstance(height, (int, str)):
            native_width = int(width)
            native_height = int(height)
        else:
            logger.warning("Failed to probe stream, using default 1920x1080")
            native_width, native_height = 1920, 1080
        logger.info("Camera resolution: %sx%s", native_width, native_height)

        if self.detect_rtsp_url != self.rtsp_url:
            info = self._probe_stream_info(self.detect_rtsp_url)
            if info and info.get("width") and info.get("height"):
                logger.info(
                    "Motion RTSP stream (%s) available: %s (%sx%s @ %s)",
                    self._detect_rtsp_url_source,
                    self._redact_rtsp_url(self.detect_rtsp_url),
                    info.get("width"),
                    info.get("height"),
                    info.get("avg_frame_rate"),
                )
            else:
                if self.detect_fallback_attempts > 0:
                    logger.warning(
                        "Motion RTSP stream (%s) did not probe cleanly; falling back to main RTSP stream",
                        self._detect_rtsp_url_source,
                    )
                    self._activate_detect_fallback(self._clock.now())
                else:
                    logger.warning(
                        "Motion RTSP stream (%s) did not probe cleanly; fallback disabled",
                        self._detect_rtsp_url_source,
                    )

        logger.info("Motion detection: 320x240@10fps (downscaled for efficiency)")

    def _handle_frame_timeout(self) -> bool:
        pipe_status = self._frame_pipeline.exit_code()
        if pipe_status is not None:
            logger.error(
                "Frame pipeline exited (code: %s). Check logs: %s/frame_pipeline.log",
                pipe_status,
                self.output_dir,
            )
        else:
            logger.warning(
                "No frames received for %.1fs (stall). Last frame %.1fs ago.",
                self.frame_timeout_s,
                self._clock.now() - self.last_successful_frame,
            )

        aggressive = self.recording_process is None
        return self._reconnect_frame_pipeline(aggressive=aggressive)

    def _stall_grace_remaining(self, now: float) -> float:
        if not self.recording_process or self.last_motion_time is None:
            return 0.0
        return max(0.0, self.motion_stop_delay - (now - self.last_motion_time))

    def _enter_stalled(self, now: float, remaining: float) -> None:
        self._stall_grace_until = now + remaining
        logger.warning(
            "Frame pipeline stalled; keeping recording alive for %.1fs",
            remaining,
        )

    def _handle_stalled_wait(self, now: float) -> bool:
        if self._stall_grace_until is None or now >= self._stall_grace_until:
            return False
        self._set_run_state(RTSPRunState.STALLED)
        self._clock.sleep(min(0.5, self._stall_grace_until - now))
        return True

    def _handle_reconnect_needed(self, now: float) -> tuple[bool, bool]:
        self._set_run_state(RTSPRunState.RECONNECTING)
        if self._handle_frame_timeout():
            return True, False
        remaining = self._stall_grace_remaining(now)
        if remaining > 0:
            self._enter_stalled(now, remaining)
            self._handle_stalled_wait(now)
            return True, True
        return False, False

    def _handle_heartbeat(
        self,
        now: float,
        frame_count: int,
        last_heartbeat: float,
    ) -> tuple[bool, bool, float]:
        if now - last_heartbeat <= self.heartbeat_s:
            return True, False, last_heartbeat

        logger.debug(
            "[HEARTBEAT] Processed %s frames, recording=%s",
            frame_count,
            self.recording_process is not None,
        )
        if not self._frame_pipeline.is_running():
            logger.error(
                "Frame pipeline died! Exit code: %s",
                self._frame_pipeline.exit_code(),
            )
            ok, stalled = self._handle_reconnect_needed(now)
            if not ok:
                return False, False, last_heartbeat
            return True, stalled, now

        return True, False, now

    def _handle_missing_dimensions(self, now: float) -> bool:
        logger.warning("Frame pipeline missing dimensions; reconnecting")
        ok, _ = self._handle_reconnect_needed(now)
        return ok

    def _recording_motion_threshold(self) -> float:
        if self.recording_process:
            return self._keepalive_threshold()
        return self.min_changed_pct

    def _handle_motion_detected(self, now: float, frame_count: int) -> None:
        logger.debug(
            "[MOTION DETECTED at frame %s] changed_pct=%.3f%%",
            frame_count,
            self._motion_detector.last_changed_pct,
        )
        self.last_motion_time = now
        if not self.recording_process:
            logger.debug("-> Starting new recording...")
            self.start_recording()
            if not self.recording_process:
                logger.error("-> Recording failed to start!")

    def _update_recording_state(self, now: float) -> None:
        if self.recording_process and self.last_motion_time is not None:
            if now - self.last_motion_time > self.motion_stop_delay:
                logger.info(
                    "No motion for %.1fs > stop_delay=%.1fs (last changed_pct=%.3f%%), stopping",
                    now - self.last_motion_time,
                    self.motion_stop_delay,
                    self._motion_detector.last_changed_pct,
                )
                self.stop_recording()
                self.last_motion_time = None
                self._set_run_state(RTSPRunState.IDLE)
                return

            self._rotate_recording_if_needed()
            self._set_run_state(RTSPRunState.RECORDING)
            return

        self._set_run_state(RTSPRunState.IDLE)

    def _process_frame(
        self,
        frame: npt.NDArray[np.uint8],
        now: float,
        frame_count: int,
    ) -> None:
        threshold = self._recording_motion_threshold()
        motion_detected = self.detect_motion(frame, threshold=threshold)

        if motion_detected:
            self._handle_motion_detected(now, frame_count)

        if self.recording_process or self.last_motion_time is not None:
            self._ensure_recording(now)

        self._update_recording_state(now)

    # State flow (simplified):
    # Idle -> Recording (motion)
    # Recording -> Stalled (frame timeout) -> Reconnecting -> Recording/Idle
    # Recording -> Idle (no motion for stop_delay)
    def _run(self) -> None:
        """Start monitoring camera for motion and recording videos."""
        self._log_startup_info()

        try:
            try:
                self._start_frame_pipeline()
            except Exception as exc:
                logger.warning("Initial frame pipeline start failed: %s", exc, exc_info=True)
                if not self._reconnect_frame_pipeline(aggressive=True):
                    return

            logger.info("Connected, monitoring for motion...")
            self.reconnect_count = 0
            self._set_run_state(RTSPRunState.IDLE)

            frame_count = 0
            last_heartbeat = self._clock.now()

            while not self._stop_event.is_set():
                now = self._clock.now()
                if self._handle_stalled_wait(now):
                    continue

                ok, stalled, last_heartbeat = self._handle_heartbeat(
                    now,
                    frame_count,
                    last_heartbeat,
                )
                if not ok:
                    break
                if stalled:
                    continue

                if self._maybe_recover_detect_stream(now):
                    self._set_run_state(RTSPRunState.RECONNECTING)
                    continue

                frame_count += 1

                raw_frame = self._frame_pipeline.read_frame(timeout_s=self.frame_timeout_s)
                if raw_frame is None:
                    now = self._clock.now()
                    if self._handle_stalled_wait(now):
                        continue

                    ok, _ = self._handle_reconnect_needed(now)
                    if not ok:
                        break

                    continue

                self._stall_grace_until = None
                frame_width = self._frame_pipeline.frame_width
                frame_height = self._frame_pipeline.frame_height
                if frame_width is None or frame_height is None:
                    if not self._handle_missing_dimensions(now):
                        break
                    continue

                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                    (frame_height, frame_width)
                )
                now = self._clock.now()
                self._process_frame(frame, now, frame_count)

        except Exception as e:
            logger.exception("Unexpected error: %s", e)
        finally:
            self.cleanup()
