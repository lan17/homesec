"""Shared helpers for setup-only probe implementations."""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from homesec.models.setup import TestConnectionResponse
from homesec.onvif.service import DEFAULT_ONVIF_PORT
from homesec.sources.rtsp.core import RTSPSourceConfig
from homesec.sources.rtsp.url_derivation import derive_detect_rtsp_url

SETUP_TEST_CAMERA_NAME = "setup-test-camera"
ONVIF_TEST_CONNECTION_TIMEOUT_CAP_S = 30.0
RTSP_TEST_CONNECTION_TIMEOUT_S = 10.0
RTSP_PREFLIGHT_COMMAND_TIMEOUT_CAP_S = 8.0
FTP_TEST_CONNECTION_TIMEOUT_S = 5.0
ONVIF_TEST_CONNECTION_TIMEOUT_S = 15.0


class OnvifSetupProbeConfig(BaseModel):
    """Validated payload for setup-time ONVIF connectivity probes."""

    host: str = Field(min_length=1)
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)
    port: int = Field(default=DEFAULT_ONVIF_PORT, ge=1, le=65535)
    timeout_s: float = Field(
        default=ONVIF_TEST_CONNECTION_TIMEOUT_S,
        gt=0.0,
        le=ONVIF_TEST_CONNECTION_TIMEOUT_CAP_S,
    )


def format_validation_error(exc: ValidationError) -> str:
    """Return a concise single-error validation message for setup UX."""
    errors = exc.errors(include_url=False)
    if not errors:
        return "Configuration validation failed."
    first = errors[0]
    loc_parts = [str(part) for part in first.get("loc", []) if str(part)]
    location = ".".join(loc_parts)
    message = str(first.get("msg", "invalid value"))
    if location:
        return f"Configuration validation failed at {location}: {message}"
    return f"Configuration validation failed: {message}"


def build_test_connection_response(
    *,
    success: bool,
    message: str,
    start: float | None = None,
    details: dict[str, object] | None = None,
) -> TestConnectionResponse:
    """Build a setup test-connection response with optional latency metadata."""
    latency_ms = None
    if start is not None:
        latency_ms = (time.perf_counter() - start) * 1000.0
    return TestConnectionResponse(
        success=success,
        message=message,
        latency_ms=latency_ms,
        details=details,
    )


def nearest_existing_parent(path: Path) -> Path | None:
    """Return the closest existing ancestor for a path, if any."""
    current = path
    while not current.exists():
        if current == current.parent:
            return None
        current = current.parent
    return current


def probe_tcp_bind(host: str, port: int) -> int:
    """Bind a temporary TCP socket and return the bound port."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return int(sock.getsockname()[1])


def resolve_rtsp_primary_url(config: RTSPSourceConfig) -> str | None:
    """Resolve the primary RTSP URL from config or env indirection."""
    if config.rtsp_url_env:
        env_url = os.getenv(config.rtsp_url_env)
        if env_url:
            return env_url
    return config.rtsp_url


def resolve_rtsp_detect_url(config: RTSPSourceConfig, primary_url: str) -> str:
    """Resolve or derive the RTSP URL used for motion-stream detection."""
    if config.detect_rtsp_url_env:
        env_url = os.getenv(config.detect_rtsp_url_env)
        if env_url:
            return env_url
    if config.detect_rtsp_url:
        return config.detect_rtsp_url
    derived = derive_detect_rtsp_url(primary_url)
    if derived is not None:
        return derived.url
    return primary_url
