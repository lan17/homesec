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
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from homesec.models.clip import Clip
from homesec.models.source import RTSPSourceConfig
from homesec.sources.base import ThreadedClipSource

logger = logging.getLogger(__name__)


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


class RTSPSource(ThreadedClipSource):
    """RTSP clip source with motion detection.

    Uses ffmpeg for frame extraction and recording; detects motion from
    downscaled grayscale frames and emits clips when recordings finish.
    """

    def __init__(self, config: RTSPSourceConfig, camera_name: str) -> None:
        """Initialize RTSP source."""
        super().__init__()
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
        self.blur_kernel = self._normalize_blur_kernel(config.blur_kernel)
        self.motion_stop_delay = float(config.stop_delay)
        self.max_recording_s = float(config.max_recording_s)
        self.max_reconnect_attempts = int(config.max_reconnect_attempts)
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
        self._stall_grace_until: float | None = None

        self._prev_motion_frame: npt.NDArray[np.uint8] | None = None
        self._last_changed_pct = 0.0
        self._last_changed_pixels = 0
        self._debug_frame_count = 0

        self.frame_pipe: subprocess.Popen[bytes] | None = None
        self._frame_pipe_stderr: Any | None = None
        self._frame_reader_thread: Thread | None = None
        self._frame_reader_stop: Event | None = None
        self._frame_queue: Queue[bytes] | None = None
        self._frame_width: int | None = None
        self._frame_height: int | None = None
        self._frame_size: int | None = None
        self.reconnect_count = 0
        self.last_successful_frame = self._last_heartbeat

        logger.info(
            "RTSPSource initialized: camera=%s, output_dir=%s",
            self.camera_name,
            self.output_dir,
        )

    def is_healthy(self) -> bool:
        """Check if source is healthy."""
        if not self._thread_is_healthy():
            return False

        age = time.monotonic() - self.last_successful_frame
        return age < (self.frame_timeout_s * 3)

    def _touch_heartbeat(self) -> None:
        self.last_successful_frame = time.monotonic()
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
            "blur_kernel": self.blur_kernel,
            "stop_delay": self.motion_stop_delay,
            "max_recording_s": self.max_recording_s,
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

    def _probe_stream_info(self, rtsp_url: str) -> dict[str, object] | None:
        cmd = [
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
            rtsp_url,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
            data = json.loads(result.stdout)
            if not data.get("streams"):
                return None
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

    def _redact_rtsp_url(self, url: str) -> str:
        if "://" not in url:
            return url
        scheme, rest = url.split("://", 1)
        if "@" not in rest:
            return url
        _creds, host = rest.split("@", 1)
        return f"{scheme}://***:***@{host}"

    def _format_cmd(self, cmd: list[str]) -> str:
        try:
            return shlex.join([str(x) for x in cmd])
        except Exception as exc:
            logger.warning("Failed to format command with shlex.join: %s", exc, exc_info=True)
            return " ".join([str(x) for x in cmd])

    def detect_motion(self, frame: npt.NDArray[np.uint8]) -> bool:
        """Return True if motion detected in frame."""
        if frame.ndim == 3:
            gray = cast(npt.NDArray[np.uint8], cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            gray = frame

        if self.blur_kernel > 1:
            gray = cast(
                npt.NDArray[np.uint8],
                cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0),
            )

        if self._prev_motion_frame is None:
            self._prev_motion_frame = gray
            self._last_changed_pct = 0.0
            self._last_changed_pixels = 0
            return False

        diff = cv2.absdiff(self._prev_motion_frame, gray)
        _, mask = cv2.threshold(diff, self.pixel_threshold, 255, cv2.THRESH_BINARY)
        changed_pixels = int(cv2.countNonZero(mask))

        total_pixels = int(gray.shape[0]) * int(gray.shape[1])
        changed_pct = (changed_pixels / total_pixels * 100.0) if total_pixels else 0.0

        self._prev_motion_frame = gray
        self._last_changed_pct = changed_pct
        self._last_changed_pixels = changed_pixels

        motion = changed_pct >= self.min_changed_pct

        if self.debug_motion:
            self._debug_frame_count += 1
            if self._debug_frame_count % 100 == 0:
                logger.debug(
                    "Motion check: changed_pct=%.3f%% changed_px=%s pixel_threshold=%s min_changed_pct=%.3f%% blur=%s",
                    changed_pct,
                    changed_pixels,
                    self.pixel_threshold,
                    self.min_changed_pct,
                    self.blur_kernel,
                )

        return motion

    def check_recording_health(self) -> bool:
        """Check if recording process is still alive."""
        if self.recording_process and self.recording_process.poll() is not None:
            proc = self.recording_process
            output_file = self.output_file
            exit_code = proc.returncode
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

        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file, stderr_log = self._make_recording_paths(timestamp)

        proc = self._spawn_recording_process(output_file, stderr_log)
        if not proc:
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
        self.recording_start_time = time.monotonic()
        self.recording_start_wall = datetime.now()
        self._recording_id = output_file.name

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

    def _spawn_recording_process(
        self, output_file: Path, stderr_log: Path
    ) -> subprocess.Popen[bytes] | None:
        cmd = [
            "ffmpeg",
            "-rtsp_transport",
            "tcp",
            "-rtsp_flags",
            "prefer_tcp",
            "-user_agent",
            "Lavf",
            "-i",
            self.rtsp_url,
            "-c",
            "copy",
            "-f",
            "mp4",
            "-y",
        ]

        user_flags = self.ffmpeg_flags

        # Naive check to see if user overrode defaults
        # If user supplies ANY -loglevel, we don't add ours.
        # If user supplies ANY -fflags, we don't add ours (to avoid concatenation complexity).
        # This allows full user control.
        has_loglevel = any(x == "-loglevel" for x in user_flags)
        if not has_loglevel:
            cmd.extend(["-loglevel", "warning"])

        has_fflags = any(x == "-fflags" for x in user_flags)
        if not has_fflags:
            cmd.extend(["-fflags", "+genpts+igndts"])

        has_fps_mode = any(x == "-fps_mode" or x == "-vsync" for x in user_flags)
        if not has_fps_mode:
            cmd.extend(["-vsync", "0"])

        # Add user flags last so they can potentially override or add to the above
        cmd.extend(user_flags)

        cmd.extend([str(output_file)])

        safe_cmd = list(cmd)
        try:
            idx = safe_cmd.index("-i")
            safe_cmd[idx + 1] = self._redact_rtsp_url(str(safe_cmd[idx + 1]))
        except Exception as exc:
            logger.warning("Failed to redact recording RTSP URL: %s", exc, exc_info=True)
        logger.debug("Recording ffmpeg: %s", self._format_cmd(safe_cmd))

        try:
            with open(stderr_log, "w") as stderr_file:
                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr_file)

            time.sleep(0.5)
            if proc.poll() is not None:
                logger.error("Recording process died immediately (exit code: %s)", proc.returncode)
                logger.error("Check logs at: %s", stderr_log)
                try:
                    with open(stderr_log) as f:
                        error_lines = f.read()
                    if error_lines:
                        logger.error("Error output: %s", error_lines[:500])
                except Exception:
                    logger.exception("Failed reading recording log: %s", stderr_log)
                return None

            return proc
        except Exception:
            logger.exception("Failed to start recording")
            return None

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
            duration_s = (time.monotonic() - started_at) if started_at else None
            logger.info(
                "Recording stopped",
                extra=self._event_extra(
                    "recording_stop",
                    recording_id=output_file.name,
                    recording_path=str(output_file),
                    duration_s=duration_s,
                    last_changed_pct=getattr(self, "_last_changed_pct", None),
                    last_changed_pixels=getattr(self, "_last_changed_pixels", None),
                ),
            )

    def _stop_recording_process(
        self, proc: subprocess.Popen[bytes], output_file: Path | None
    ) -> None:
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

    def _rotate_recording_if_needed(self) -> None:
        if not self.recording_process or not self.recording_start_time:
            return

        now = time.monotonic()
        if (now - self.recording_start_time) < self.max_recording_s:
            return

        if not self.last_motion_time:
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

        new_proc = self._spawn_recording_process(new_output, new_log)
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

    def probe_stream_resolution(self) -> tuple[int, int]:
        """Probe RTSP stream to get native resolution."""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            "-rtsp_transport",
            "tcp",
            self.rtsp_url,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
            data = json.loads(result.stdout)
            width = data["streams"][0]["width"]
            height = data["streams"][0]["height"]
            return int(width), int(height)
        except Exception as e:
            logger.warning(
                "Failed to probe stream, using default 1920x1080: %s",
                e,
                exc_info=True,
            )
            return 1920, 1080

    def get_frame_pipe(self) -> tuple[subprocess.Popen[bytes], Any, int, int]:
        """Create ffmpeg process to get frames for motion detection."""
        detect_width, detect_height = 320, 240

        stderr_log = self.output_dir / "frame_pipeline.log"
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        if self.hwaccel_config.is_available and not self._hwaccel_failed:
            hwaccel = self.hwaccel_config.hwaccel
            if hwaccel is not None:
                cmd.extend(["-hwaccel", hwaccel])
            if self.hwaccel_config.hwaccel_device:
                cmd.extend(["-hwaccel_device", self.hwaccel_config.hwaccel_device])
        elif self._hwaccel_failed:
            logger.info("Hardware acceleration disabled due to previous failures")

        # 2. Global Flags (Robustness & Logging)
        user_flags = self.ffmpeg_flags

        has_loglevel = any(x == "-loglevel" for x in user_flags)
        if not has_loglevel:
            cmd.extend(["-loglevel", "warning"])

        has_fflags = any(x == "-fflags" for x in user_flags)
        if not has_fflags:
            cmd.extend(["-fflags", "+genpts+igndts"])

        # Add all user flags to global scope.
        # Users who want input-specific flags (before -i) must rely on ffmpeg parsing them correctly
        # or we would need a more complex config structure.
        # For now, most robustness flags (-re, -rtsp_transport, etc) work as global or are handled below.
        cmd.extend(user_flags)

        base_input_args = [
            "-rtsp_transport",
            "tcp",
            "-i",
            self.detect_rtsp_url,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-vf",
            f"fps=10,scale={detect_width}:{detect_height}",
            "-an",
            "-",
        ]

        timeout_us_connect = str(int(max(0.1, self.rtsp_connect_timeout_s) * 1_000_000))
        attempts: list[tuple[str, list[str]]] = [
            (
                "stimeout",
                ["-stimeout", timeout_us_connect] + base_input_args,
            ),
            ("stimeout", ["-stimeout", timeout_us_connect] + base_input_args),
            ("no_timeouts", base_input_args),
        ]

        process: subprocess.Popen[bytes] | None = None
        stderr_file: Any | None = None

        for label, extra_args in attempts:
            cmd_attempt = list(cmd) + extra_args
            logger.debug("Starting frame pipeline (%s), logging to: %s", label, stderr_log)
            safe_cmd = list(cmd_attempt)
            try:
                idx = safe_cmd.index("-i")
                safe_cmd[idx + 1] = self._redact_rtsp_url(str(safe_cmd[idx + 1]))
            except Exception as exc:
                logger.warning(
                    "Failed to redact frame pipeline RTSP URL: %s",
                    exc,
                    exc_info=True,
                )
            logger.debug("Frame pipeline ffmpeg (%s): %s", label, self._format_cmd(safe_cmd))

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

            time.sleep(1)
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

    def _stop_frame_pipeline(self) -> None:
        if self._frame_reader_stop is not None:
            self._frame_reader_stop.set()

        if self.frame_pipe is not None:
            try:
                self._stop_process(self.frame_pipe, "Frame pipeline", terminate_timeout_s=2)
            except Exception:
                logger.exception("Error stopping frame pipeline process")
            self.frame_pipe = None

        if self._frame_reader_thread is not None:
            try:
                self._frame_reader_thread.join(timeout=5)
            except Exception:
                logger.exception("Error joining frame reader thread")
            self._frame_reader_thread = None

        self._frame_reader_stop = None

        if self._frame_pipe_stderr is not None:
            try:
                self._frame_pipe_stderr.close()
            except Exception:
                logger.exception("Error closing frame pipeline stderr log")
            self._frame_pipe_stderr = None

        self._frame_queue = None
        self._frame_width = None
        self._frame_height = None
        self._frame_size = None

    def _start_frame_pipeline(self) -> None:
        self._stop_frame_pipeline()

        self.frame_pipe, self._frame_pipe_stderr, self._frame_width, self._frame_height = (
            self.get_frame_pipe()
        )
        self._frame_size = int(self._frame_width) * int(self._frame_height)
        self._frame_queue = Queue(maxsize=self.frame_queue_size)
        self._frame_reader_stop = Event()

        frame_pipe = self.frame_pipe
        frame_size = self._frame_size
        frame_queue = self._frame_queue
        stop_event = self._frame_reader_stop

        def reader_loop() -> None:
            stdout = frame_pipe.stdout if frame_pipe else None
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

                self._touch_heartbeat()

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

        self._frame_reader_thread = Thread(target=reader_loop, name="frame-reader", daemon=True)
        self._frame_reader_thread.start()

    def _wait_for_first_frame(self, timeout_s: float) -> bool:
        if not self._frame_queue:
            return False
        try:
            raw = self._frame_queue.get(timeout=float(timeout_s))
        except Empty:
            return False
        try:
            self._frame_queue.put_nowait(raw)
        except Full:
            pass
        return True

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
                )
            else:
                logger.warning(
                    "Reconnect attempt %s/%s (mode=%s)...",
                    self.reconnect_count,
                    self.max_reconnect_attempts,
                    "aggressive" if aggressive else "normal",
                )
                if self.reconnect_count >= self.max_reconnect_attempts:
                    logger.error("Max reconnect attempts reached. Camera may be offline.")
                    return False

            if first_attempt and aggressive:
                sleep_s = 0.0
            else:
                jitter = random.uniform(0.0, min(0.25, backoff_s * 0.25))
                sleep_s = backoff_s + jitter
            first_attempt = False

            if sleep_s > 0:
                logger.debug("Waiting %.2fs before reconnect...", sleep_s)
                time.sleep(sleep_s)

            try:
                self._start_frame_pipeline()
            except Exception as exc:
                logger.warning("Failed to restart frame pipeline: %s", exc, exc_info=True)
                if aggressive:
                    backoff_s = min(backoff_s * 1.6, backoff_cap_s)
                continue

            startup_timeout_s = min(2.0, max(0.5, float(self.frame_timeout_s)))
            if self._wait_for_first_frame(startup_timeout_s):
                logger.info("Reconnected successfully")
                self.reconnect_count = 0
                return True

            logger.warning("Frame pipeline restarted but still no frames; retrying...")
            try:
                self._stop_frame_pipeline()
            except Exception as exc:
                logger.warning(
                    "Failed to stop frame pipeline after reconnect: %s",
                    exc,
                    exc_info=True,
                )
            if aggressive:
                backoff_s = min(backoff_s * 1.6, backoff_cap_s)

        return False

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
        logger.info("Max reconnect attempts: %s", self.max_reconnect_attempts)
        if self.max_reconnect_attempts == 0:
            logger.info("Reconnect policy: retry forever")

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
        native_width, native_height = self.probe_stream_resolution()
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
                logger.warning(
                    "Motion RTSP stream (%s) did not probe cleanly; falling back to main RTSP stream",
                    self._detect_rtsp_url_source,
                )
                self.detect_rtsp_url = self.rtsp_url
                self._detect_rtsp_url_source = "fallback_to_rtsp_url"

        logger.info("Motion detection: 320x240@10fps (downscaled for efficiency)")

    def _handle_frame_timeout(self) -> bool:
        pipe_status = self.frame_pipe.poll() if self.frame_pipe else None
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
                time.monotonic() - self.last_successful_frame,
            )

        aggressive = self.recording_process is None
        return self._reconnect_frame_pipeline(aggressive=aggressive)

    def _stall_grace_remaining(self, now: float) -> float:
        if not self.recording_process or not self.last_motion_time:
            return 0.0
        return max(0.0, self.motion_stop_delay - (now - self.last_motion_time))

    def _run(self) -> None:
        """Start monitoring camera for motion and recording videos."""
        self._log_startup_info()

        try:
            self._start_frame_pipeline()
            logger.info("Connected, monitoring for motion...")
            self.reconnect_count = 0

            frame_count = 0
            last_heartbeat = time.monotonic()

            while not self._stop_event.is_set():
                if time.monotonic() - last_heartbeat > self.heartbeat_s:
                    logger.debug(
                        "[HEARTBEAT] Processed %s frames, recording=%s",
                        frame_count,
                        self.recording_process is not None,
                    )
                    if self.frame_pipe and self.frame_pipe.poll() is not None:
                        logger.error(
                            "Frame pipeline died! Exit code: %s", self.frame_pipe.returncode
                        )
                        break
                    last_heartbeat = time.monotonic()

                frame_count += 1

                if (
                    not self._frame_queue
                    or not self._frame_width
                    or not self._frame_height
                    or not self._frame_size
                ):
                    break

                try:
                    raw_frame = self._frame_queue.get(timeout=self.frame_timeout_s)
                except Empty:
                    now = time.monotonic()
                    if self._stall_grace_until is not None and now < self._stall_grace_until:
                        time.sleep(min(0.5, self._stall_grace_until - now))
                        continue

                    if not self._handle_frame_timeout():
                        remaining = self._stall_grace_remaining(now)
                        if remaining > 0:
                            self._stall_grace_until = now + remaining
                            logger.warning(
                                "Frame pipeline stalled; keeping recording alive for %.1fs",
                                remaining,
                            )
                            time.sleep(min(0.5, remaining))
                            continue
                        break
                    continue

                self._stall_grace_until = None
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                    (self._frame_height, self._frame_width)
                )
                now = time.monotonic()
                motion_detected = self.detect_motion(frame)

                if motion_detected:
                    logger.debug(
                        "[MOTION DETECTED at frame %s] changed_pct=%.3f%%",
                        frame_count,
                        self._last_changed_pct,
                    )
                    self.last_motion_time = now
                    if not self.recording_process:
                        logger.debug("-> Starting new recording...")
                        self.start_recording()
                        if not self.recording_process:
                            logger.error("-> Recording failed to start!")
                    else:
                        if not self.check_recording_health():
                            logger.warning("-> Recording was dead, trying to restart...")
                            self.recording_process = None
                            self.start_recording()

                if self.recording_process and self.last_motion_time:
                    keepalive_threshold = self.min_changed_pct * 0.5
                    if self._last_changed_pct >= keepalive_threshold:
                        self.last_motion_time = now

                if self.recording_process and self.last_motion_time:
                    if not self.check_recording_health():
                        logger.warning("Recording died, stopping...")
                        self.stop_recording()
                        self.last_motion_time = None
                    elif now - self.last_motion_time > self.motion_stop_delay:
                        logger.info(
                            "No motion for %.1fs > stop_delay=%.1fs (last changed_pct=%.3f%%), stopping",
                            now - self.last_motion_time,
                            self.motion_stop_delay,
                            self._last_changed_pct,
                        )
                        self.stop_recording()
                        self.last_motion_time = None
                    else:
                        self._rotate_recording_if_needed()

        except Exception as e:
            logger.exception("Unexpected error: %s", e)
        finally:
            self.cleanup()
