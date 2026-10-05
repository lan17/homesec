"""Source-private motion input contract and the existing FFmpeg/OpenCV adapter."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from homesec.sources.rtsp.frame_pipeline import FfmpegFramePipeline, FramePipeline
from homesec.sources.rtsp.motion import MotionDetector
from homesec.sources.rtsp.recording_profile import MotionProfile


class MotionObservation(BaseModel):
    """One bounded observation from a consumed detection frame."""

    model_config = ConfigDict(extra="forbid", strict=True)

    motion: bool
    changed_pixels: int = Field(ge=0, le=320 * 240)
    changed_pct: float = Field(ge=0.0, le=100.0, allow_inf_nan=False)


class MotionInput(Protocol):
    """Motion input lifecycle independent of recording and preview consumers."""

    def start(self, rtsp_url: str) -> None: ...

    def stop(self) -> None: ...

    def is_running(self) -> bool: ...

    def exit_code(self) -> int | None: ...

    def read_motion(self, timeout_s: float, threshold: float) -> MotionObservation | None: ...

    def discard_frame(self, timeout_s: float) -> bool: ...

    def set_motion_profile(self, profile: MotionProfile) -> None: ...


class FfmpegMotionInput:
    """Detect only frames consumed after the existing bounded raw-frame queue."""

    def __init__(self, frame_pipeline: FramePipeline, detector: MotionDetector) -> None:
        self._frame_pipeline = frame_pipeline
        self._detector = detector

    def start(self, rtsp_url: str) -> None:
        self._frame_pipeline.start(rtsp_url)
        self._detector.reset()

    def stop(self) -> None:
        self._frame_pipeline.stop()
        self._detector.reset()

    def is_running(self) -> bool:
        return self._frame_pipeline.is_running()

    def exit_code(self) -> int | None:
        return self._frame_pipeline.exit_code()

    def read_motion(self, timeout_s: float, threshold: float) -> MotionObservation | None:
        raw = self._frame_pipeline.read_frame(timeout_s)
        if raw is None:
            return None
        width = self._frame_pipeline.frame_width
        height = self._frame_pipeline.frame_height
        if width is None or height is None:
            return None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
        motion = self._detector.detect(frame, threshold=threshold)
        return MotionObservation(
            motion=motion,
            changed_pixels=self._detector.last_changed_pixels,
            changed_pct=self._detector.last_changed_pct,
        )

    def discard_frame(self, timeout_s: float) -> bool:
        # Reconnect readiness consumes a frame without initializing the detector.
        return self._frame_pipeline.read_frame(timeout_s) is not None

    def set_motion_profile(self, profile: MotionProfile) -> None:
        if isinstance(self._frame_pipeline, FfmpegFramePipeline):
            self._frame_pipeline.set_motion_profile(profile)
