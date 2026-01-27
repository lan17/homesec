"""FTP source configuration model."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class FtpSourceConfig(BaseModel):
    """FTP source configuration."""

    model_config = {"extra": "forbid"}

    camera_name: str | None = Field(
        default=None,
        description="Optional human-friendly camera name.",
    )
    host: str = Field(
        default="0.0.0.0",
        description="FTP bind address.",
    )
    port: int = Field(
        default=2121,
        ge=0,
        le=65535,
        description="FTP listen port (0 lets the OS choose an ephemeral port).",
    )
    root_dir: str = Field(
        default="./ftp_incoming",
        description="FTP root directory for uploads.",
    )
    ftp_subdir: str | None = Field(
        default=None,
        description="Optional subdirectory under root_dir.",
    )
    anonymous: bool = Field(
        default=True,
        description="Allow anonymous FTP uploads.",
    )
    username_env: str | None = Field(
        default=None,
        description="Environment variable containing FTP username.",
    )
    password_env: str | None = Field(
        default=None,
        description="Environment variable containing FTP password.",
    )
    perms: str = Field(
        default="elw",
        description="pyftpdlib permissions string.",
    )
    passive_ports: str | None = Field(
        default=None,
        description="Passive ports range (e.g., '60000-60100' or '60000,60010').",
    )
    masquerade_address: str | None = Field(
        default=None,
        description="Optional masquerade address for passive mode.",
    )
    heartbeat_s: float = Field(
        default=30.0,
        ge=0.0,
        description="Seconds between FTP health checks.",
    )
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".mp4"],
        description="Allowed file extensions for uploaded clips.",
    )
    delete_non_matching: bool = Field(
        default=True,
        description="Delete files with disallowed extensions.",
    )
    delete_incomplete: bool = Field(
        default=True,
        description="Delete incomplete uploads when enabled.",
    )
    default_duration_s: float = Field(
        default=10.0,
        ge=0.0,
        description="Fallback clip duration when timestamps are missing.",
    )
    log_level: str = Field(
        default="INFO",
        description="FTP server log level.",
    )

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
