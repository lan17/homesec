from __future__ import annotations

import logging
import platform
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from homesec.sources.rtsp.inspection import (
    CapabilityEvidence,
    DecodeEvidence,
    InspectionFailure,
    InspectionGroup,
)

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


@dataclass(frozen=True, slots=True)
class H264HardwareEncoder:
    """Preview H.264 encoder derived from an ffmpeg hardware backend."""

    ffmpeg_encoder: str
    ffmpeg_global_args: tuple[str, ...] = ()
    requires_hwupload: bool = False


class HardwareAccelDetector:
    """Inspect the linked build and require decoded frames on the requested device."""

    @staticmethod
    def detect(rtsp_url: str, *, helper_path: str = "homesec-webrtc") -> HardwareAccelConfig:
        try:
            with InspectionGroup(helper_path=helper_path, total_timeout_s=46.0) as inspection:
                for backend, device in HardwareAccelDetector._candidates():
                    capabilities = inspection.request(
                        "inspect_capabilities",
                        timeout_s=2.0,
                        backend=backend,
                        device=device,
                    )
                    if not isinstance(capabilities, CapabilityEvidence):
                        raise InspectionFailure("helper_failure")
                    if capabilities.device is None or not capabilities.device.available:
                        continue
                    try:
                        evidence = inspection.request(
                            "inspect_decode",
                            timeout_s=10.0,
                            rtsp_url=rtsp_url,
                            backend=backend,
                            device=device,
                            sample_frames=5,
                        )
                    except InspectionFailure as exc:
                        if exc.code in {"helper_failure", "cancelled", "cleanup_failed"}:
                            raise
                        logger.info(
                            "Native hardware decode failed: backend=%s reason=%s", backend, exc.code
                        )
                        continue
                    if (
                        isinstance(evidence, DecodeEvidence)
                        and evidence.backend == backend
                        and evidence.frames_decoded >= 5
                    ):
                        return HardwareAccelConfig(hwaccel=backend, hwaccel_device=device)
        except InspectionFailure as exc:
            logger.info("Native hardware validation unavailable: %s", exc.code)
        return HardwareAccelConfig(hwaccel=None)

    @staticmethod
    def _candidates() -> list[tuple[str, str | None]]:
        candidates: list[tuple[str, str | None]] = []
        if Path("/dev/dri/renderD128").exists():
            candidates.append(("vaapi", "/dev/dri/renderD128"))
        candidates.append(("cuda", None))
        if platform.system() == "Darwin":
            candidates.append(("videotoolbox", None))
        candidates.append(("qsv", None))
        return candidates

    @staticmethod
    def select_h264_encoder(
        config: HardwareAccelConfig,
        *,
        helper_path: str = "homesec-webrtc",
        inspection: InspectionGroup | None = None,
    ) -> H264HardwareEncoder | None:
        candidate = HardwareAccelDetector._candidate_h264_encoder(config)
        if candidate is None:
            return None
        try:
            context = (
                nullcontext(inspection)
                if inspection is not None
                else InspectionGroup(helper_path=helper_path, total_timeout_s=3.0)
            )
            with context as owner:
                evidence = owner.request("inspect_capabilities", timeout_s=2.0)
                if (
                    isinstance(evidence, CapabilityEvidence)
                    and candidate.ffmpeg_encoder in evidence.encoders
                ):
                    return candidate
        except InspectionFailure:
            pass
        return None

    @staticmethod
    def _candidate_h264_encoder(config: HardwareAccelConfig) -> H264HardwareEncoder | None:
        """Map a detected hwaccel backend to the corresponding H.264 encoder."""
        hwaccel = (config.hwaccel or "").strip().lower()

        match hwaccel:
            case "vaapi":
                ffmpeg_global_args: tuple[str, ...] = ()
                if config.hwaccel_device:
                    ffmpeg_global_args = ("-vaapi_device", config.hwaccel_device)
                return H264HardwareEncoder(
                    ffmpeg_encoder="h264_vaapi",
                    ffmpeg_global_args=ffmpeg_global_args,
                    requires_hwupload=True,
                )
            case "cuda":
                return H264HardwareEncoder(ffmpeg_encoder="h264_nvenc")
            case "videotoolbox":
                return H264HardwareEncoder(ffmpeg_encoder="h264_videotoolbox")
            case "qsv":
                return H264HardwareEncoder(ffmpeg_encoder="h264_qsv")
            case _:
                return None
