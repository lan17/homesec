from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

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
