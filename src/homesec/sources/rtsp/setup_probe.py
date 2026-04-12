"""Setup-only RTSP connectivity probe."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from pathlib import Path

from pydantic import ValidationError

from homesec.models.setup import TestConnectionResponse
from homesec.plugins.registry import PluginType, validate_plugin
from homesec.services.setup_probe_support import (
    RTSP_PREFLIGHT_COMMAND_TIMEOUT_CAP_S,
    RTSP_TEST_CONNECTION_TIMEOUT_S,
    SETUP_TEST_CAMERA_NAME,
    build_test_connection_response,
    format_validation_error,
    resolve_rtsp_detect_url,
    resolve_rtsp_primary_url,
)
from homesec.services.setup_probes import setup_probe
from homesec.sources.rtsp.core import RTSPSourceConfig
from homesec.sources.rtsp.preflight import PreflightError, RTSPStartupPreflight

logger = logging.getLogger(__name__)


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
