"""Source configuration models."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class RTSPSourceConfig(BaseModel):
    """RTSP source configuration."""

    model_config = {"extra": "forbid"}

    camera_name: str | None = None
    rtsp_url_env: str | None = None
    rtsp_url: str | None = None
    detect_rtsp_url_env: str | None = None
    detect_rtsp_url: str | None = None
    output_dir: str = "./recordings"
    pixel_threshold: int = 45
    min_changed_pct: float = 1.0
    recording_sensitivity_factor: float = Field(default=2.0, gt=0)
    blur_kernel: int = 5
    stop_delay: float = 10.0
    max_recording_s: float = 60.0
    max_reconnect_attempts: int = 0
    detect_fallback_attempts: int = Field(default=3, ge=0)
    disable_hwaccel: bool = False
    frame_timeout_s: float = 2.0
    frame_queue_size: int = 20
    reconnect_backoff_s: float = 1.0
    debug_motion: bool = False
    heartbeat_s: float = 30.0
    rtsp_connect_timeout_s: float = 2.0
    rtsp_io_timeout_s: float = 2.0
    ffmpeg_flags: list[str] = Field(default_factory=list)


class LocalFolderSourceConfig(BaseModel):
    """Local folder source configuration."""

    model_config = {"extra": "forbid"}

    camera_name: str | None = None
    watch_dir: str = "recordings"
    poll_interval: float = 1.0
    stability_threshold_s: float = 3.0


class FtpSourceConfig(BaseModel):
    """FTP source configuration."""

    model_config = {"extra": "forbid"}

    camera_name: str | None = None
    host: str = "0.0.0.0"
    port: int = 2121
    root_dir: str = "./ftp_incoming"
    ftp_subdir: str | None = None
    anonymous: bool = True
    username_env: str | None = None
    password_env: str | None = None
    perms: str = "elw"
    passive_ports: str | None = None
    masquerade_address: str | None = None
    heartbeat_s: float = 30.0
    allowed_extensions: list[str] = Field(default_factory=lambda: [".mp4"])
    delete_non_matching: bool = True
    delete_incomplete: bool = True
    default_duration_s: float = 10.0
    log_level: str = "INFO"

    @field_validator("allowed_extensions")
    @classmethod
    def _normalize_extensions(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            ext = str(item).strip().lower()
            if not ext:
                continue
            if not ext.startswith("."):
                ext = f".{ext}"
            cleaned.append(ext)
        return cleaned
