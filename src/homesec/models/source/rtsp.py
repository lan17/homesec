"""RTSP source configuration models."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class RTSPMotionConfig(BaseModel):
    """Motion detection configuration."""

    model_config = {"extra": "forbid"}

    pixel_threshold: int = Field(
        default=45,
        ge=0,
        description="Pixel intensity delta required to count a pixel as changed.",
    )
    min_changed_pct: float = Field(
        default=1.0,
        ge=0.0,
        description="Percent of pixels that must change to trigger motion (idle state).",
    )
    recording_sensitivity_factor: float = Field(
        default=2.0,
        ge=1.0,
        description="Factor to reduce the threshold while recording (>=1.0).",
    )
    blur_kernel: int = Field(
        default=5,
        ge=0,
        description="Gaussian blur kernel size (odd or zero; even values are normalized).",
    )


class RTSPRecordingConfig(BaseModel):
    """Recording lifecycle configuration."""

    model_config = {"extra": "forbid"}

    stop_delay: float = Field(
        default=10.0,
        ge=0.0,
        description="Seconds to keep recording after motion stops.",
    )
    max_recording_s: float = Field(
        default=60.0,
        gt=0.0,
        description="Maximum seconds per recording before rotating.",
    )


class RTSPStreamConfig(BaseModel):
    """RTSP/ffmpeg transport configuration."""

    model_config = {"extra": "forbid"}

    connect_timeout_s: float = Field(
        default=2.0,
        ge=0.0,
        description="RTSP connect timeout (seconds) passed to ffmpeg/ffprobe when supported.",
    )
    io_timeout_s: float = Field(
        default=2.0,
        ge=0.0,
        description="RTSP I/O timeout (seconds) passed to ffmpeg/ffprobe when supported.",
    )
    ffmpeg_flags: list[str] = Field(
        default_factory=list,
        description="Additional ffmpeg flags appended to the command.",
    )
    disable_hwaccel: bool = Field(
        default=False,
        description="Disable hardware-accelerated decoding.",
    )


class RTSPReconnectConfig(BaseModel):
    """Reconnect and fallback policy."""

    model_config = {"extra": "forbid"}

    max_attempts: int = Field(
        default=0,
        ge=0,
        description="Max reconnect attempts (0 = retry forever).",
    )
    backoff_s: float = Field(
        default=1.0,
        ge=0.0,
        description="Base backoff (seconds) between reconnect attempts.",
    )
    detect_fallback_attempts: int = Field(
        default=3,
        ge=0,
        description="Failures before falling back from detect stream to main stream.",
    )


class RTSPRuntimeConfig(BaseModel):
    """Runtime loop configuration."""

    model_config = {"extra": "forbid"}

    frame_timeout_s: float = Field(
        default=2.0,
        ge=0.0,
        description="Seconds without frames before considering the pipeline stalled.",
    )
    frame_queue_size: int = Field(
        default=20,
        ge=1,
        description="Frame queue size used by the frame reader thread.",
    )
    heartbeat_s: float = Field(
        default=30.0,
        ge=0.0,
        description="Seconds between heartbeat logs.",
    )
    debug_motion: bool = Field(
        default=False,
        description="Enable verbose motion detection logging.",
    )


class RTSPSourceConfig(BaseModel):
    """RTSP source configuration."""

    model_config = {"extra": "forbid"}

    camera_name: str | None = Field(
        default=None,
        description="Optional human-friendly camera name.",
    )
    rtsp_url_env: str | None = Field(
        default=None,
        description="Environment variable containing the RTSP URL.",
    )
    rtsp_url: str | None = Field(
        default=None,
        description="RTSP URL for the main stream.",
    )
    detect_rtsp_url_env: str | None = Field(
        default=None,
        description="Environment variable containing the detect stream RTSP URL.",
    )
    detect_rtsp_url: str | None = Field(
        default=None,
        description="RTSP URL for the detect stream.",
    )
    output_dir: str = Field(
        default="./recordings",
        description="Directory to store recordings and logs.",
    )

    motion: RTSPMotionConfig = Field(default_factory=RTSPMotionConfig)
    recording: RTSPRecordingConfig = Field(default_factory=RTSPRecordingConfig)
    stream: RTSPStreamConfig = Field(default_factory=RTSPStreamConfig)
    reconnect: RTSPReconnectConfig = Field(default_factory=RTSPReconnectConfig)
    runtime: RTSPRuntimeConfig = Field(default_factory=RTSPRuntimeConfig)

    @model_validator(mode="after")
    def _require_rtsp_url(self) -> RTSPSourceConfig:
        if not (self.rtsp_url or self.rtsp_url_env):
            raise ValueError("rtsp_url_env or rtsp_url required for RTSP source")
        return self
