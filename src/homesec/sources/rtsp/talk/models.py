"""Models for RTSP/ONVIF push-to-talk adapters."""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from homesec.sources.rtsp.talk.rtsp_auth import RTSPCredentials


class TalkTransport(StrEnum):
    """Supported camera talk RTP transport modes for MVP."""

    RTSP_TCP_INTERLEAVED = "rtsp_tcp_interleaved"


class TalkCodec(StrEnum):
    """Supported camera speaker codecs for the Phase 1/MVP protocol path."""

    PCMU_8000 = "PCMU/8000"
    PCMA_8000 = "PCMA/8000"


class ONVIFBackchannelConfig(BaseModel):
    """ONVIF RTSP audio-backchannel camera configuration."""

    model_config = {"extra": "forbid"}

    rtsp_url_env: str | None = Field(
        default=None,
        description="Environment variable containing the RTSP URL used for talk.",
    )
    rtsp_url: str | None = Field(
        default=None,
        description="RTSP URL used for talk. Credentials are redacted before logging.",
    )
    username_env: str | None = Field(
        default=None,
        description="Environment variable containing the RTSP username used for talk.",
    )
    password_env: str | None = Field(
        default=None,
        description="Environment variable containing the RTSP password used for talk.",
    )
    username: str | None = Field(
        default=None,
        description="RTSP username used for manual/probe sessions.",
    )
    password: str | None = Field(
        default=None,
        description="RTSP password used for manual/probe sessions.",
    )
    preferred_codecs: list[str] = Field(
        default_factory=lambda: [TalkCodec.PCMU_8000.value, TalkCodec.PCMA_8000.value],
        min_length=1,
        description="Ordered codec preference list using SDP names such as PCMU/8000.",
    )
    transport: Literal["rtsp_tcp_interleaved"] = "rtsp_tcp_interleaved"
    connect_timeout_s: float = Field(default=5.0, gt=0.0, le=30.0)
    io_timeout_s: float = Field(default=5.0, gt=0.0, le=30.0)
    user_agent: str = "HomeSec/PushToTalk"

    @field_validator("preferred_codecs")
    @classmethod
    def _require_mvp_codec(cls, value: list[str]) -> list[str]:
        normalized = [_normalize_codec_name(item) for item in value]
        supported = {codec.value for codec in TalkCodec}
        unsupported = [item for item in normalized if item not in supported]
        if unsupported:
            raise ValueError("ONVIF talk supports only PCMU/8000 and PCMA/8000")
        return normalized

    @model_validator(mode="after")
    def _require_rtsp_url_source(self) -> ONVIFBackchannelConfig:
        if not (self.rtsp_url or self.rtsp_url_env):
            raise ValueError("rtsp_url_env or rtsp_url required for ONVIF talk backchannel")
        if (self.username is None) != (self.password is None):
            raise ValueError("username and password must be configured together")
        if (self.username_env is None) != (self.password_env is None):
            raise ValueError("username_env and password_env must be configured together")
        return self

    def resolve_rtsp_url(self, environ: Mapping[str, str] | None = None) -> str:
        """Resolve the configured RTSP URL without exposing credentials to callers."""
        if self.rtsp_url is not None:
            return self.rtsp_url
        if self.rtsp_url_env is None:
            raise ValueError("rtsp_url_env or rtsp_url required for ONVIF talk backchannel")
        env = os.environ if environ is None else environ
        try:
            return env[self.rtsp_url_env]
        except KeyError as exc:
            raise ValueError(
                f"RTSP URL environment variable is not set: {self.rtsp_url_env}"
            ) from exc

    def resolve_credentials(
        self, environ: Mapping[str, str] | None = None
    ) -> RTSPCredentials | None:
        """Resolve optional RTSP credentials from explicit values or environment variables."""
        if self.username is not None and self.password is not None:
            return RTSPCredentials(username=self.username, password=self.password)
        if self.username_env is None and self.password_env is None:
            return None
        if self.username_env is None or self.password_env is None:
            raise ValueError("username_env and password_env must be configured together")
        env = os.environ if environ is None else environ
        try:
            return RTSPCredentials(
                username=env[self.username_env],
                password=env[self.password_env],
            )
        except KeyError as exc:
            raise ValueError(
                f"RTSP credential environment variable is not set: {exc.args[0]}"
            ) from exc


def _normalize_codec_name(value: str) -> str:
    parts = value.strip().split("/")
    if len(parts) < 2:
        return value.strip().upper()
    encoding = parts[0].upper()
    clock_rate = parts[1]
    channels = parts[2] if len(parts) >= 3 else None
    if channels in (None, "", "1"):
        return f"{encoding}/{clock_rate}"
    return f"{encoding}/{clock_rate}/{channels}"
