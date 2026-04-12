"""Setup-only RTSP connectivity probe."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path

from pydantic import ValidationError

from homesec.models.setup import TestConnectionResponse
from homesec.plugins.registry import PluginType, validate_plugin
from homesec.services.setup_probe_support import (
    SETUP_TEST_CAMERA_NAME,
    build_test_connection_response,
    format_validation_error,
)
from homesec.services.setup_probes import setup_probe
from homesec.sources.rtsp.core import RTSPSourceConfig
from homesec.sources.rtsp.preflight import PreflightError, RTSPStartupPreflight
from homesec.sources.rtsp.url_derivation import derive_detect_rtsp_url

logger = logging.getLogger(__name__)

RTSP_TEST_CONNECTION_TIMEOUT_S = 10.0
RTSP_PREFLIGHT_COMMAND_TIMEOUT_CAP_S = 8.0


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


@setup_probe("camera", "rtsp", timeout_s=RTSP_TEST_CONNECTION_TIMEOUT_S + 1.0)
async def test_rtsp_camera_connection(
    *,
    config: dict[str, object],
) -> TestConnectionResponse:
    """Validate RTSP config and run bounded startup preflight."""
    start = time.perf_counter()
    try:
        validated = validate_plugin(
            PluginType.SOURCE,
            "rtsp",
            config,
            camera_name=SETUP_TEST_CAMERA_NAME,
        )
    except ValidationError as exc:
        return build_test_connection_response(
            success=False,
            message=format_validation_error(exc),
            start=start,
        )

    if not isinstance(validated, RTSPSourceConfig):
        return build_test_connection_response(
            success=False,
            message=f"Unexpected rtsp config model: {type(validated).__name__}",
            start=start,
        )

    primary_url = resolve_rtsp_primary_url(validated)
    if not primary_url:
        return build_test_connection_response(
            success=False,
            message="RTSP URL not resolved. Provide rtsp_url or set rtsp_url_env.",
            start=start,
        )
    detect_url = resolve_rtsp_detect_url(validated, primary_url)

    with tempfile.TemporaryDirectory(prefix="homesec-rtsp-probe-") as temp_dir:
        preflight = RTSPStartupPreflight(
            output_dir=Path(temp_dir),
            rtsp_connect_timeout_s=float(validated.stream.connect_timeout_s),
            rtsp_io_timeout_s=float(validated.stream.io_timeout_s),
            command_timeout_s=min(
                RTSP_TEST_CONNECTION_TIMEOUT_S,
                RTSP_PREFLIGHT_COMMAND_TIMEOUT_CAP_S,
            ),
        )
        try:
            outcome = await asyncio.wait_for(
                asyncio.to_thread(
                    preflight.run,
                    camera_name=validated.camera_name or SETUP_TEST_CAMERA_NAME,
                    primary_rtsp_url=primary_url,
                    detect_rtsp_url=detect_url,
                ),
                timeout=RTSP_TEST_CONNECTION_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return build_test_connection_response(
                success=False,
                message=f"RTSP probe timed out after {RTSP_TEST_CONNECTION_TIMEOUT_S:.1f}s.",
                start=start,
            )
        except Exception:
            logger.warning("RTSP setup test probe failed", exc_info=True)
            return build_test_connection_response(
                success=False,
                message="RTSP probe failed. Check stream URL, credentials, and network connectivity.",
                start=start,
            )

    if isinstance(outcome, PreflightError):
        return build_test_connection_response(
            success=False,
            message=outcome.message,
            start=start,
            details={
                "stage": outcome.stage,
                "camera_key": outcome.camera_key,
            },
        )

    return build_test_connection_response(
        success=True,
        message="RTSP probe succeeded.",
        start=start,
        details={
            "session_mode": outcome.diagnostics.session_mode,
            "selected_recording_profile": outcome.diagnostics.selected_recording_profile,
            "probed_streams": len(outcome.diagnostics.probes),
        },
    )
