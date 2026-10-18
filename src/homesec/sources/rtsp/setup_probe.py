"""Setup-only RTSP connectivity probe."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from homesec.models.setup import PreflightCheckResponse, TestConnectionResponse
from homesec.plugins.registry import PluginType, validate_plugin
from homesec.services.setup_probe_support import (
    SETUP_TEST_CAMERA_NAME,
    build_test_connection_response,
    format_validation_error,
)
from homesec.services.setup_probes import SetupProbeContext, setup_probe
from homesec.sources.rtsp.core import RTSPSourceConfig
from homesec.sources.rtsp.discovery import build_camera_key
from homesec.sources.rtsp.inspection import (
    CapabilityEvidence,
    InspectionFailure,
    InspectionGroup,
    select_helper_path,
)
from homesec.sources.rtsp.preflight import PreflightError, RTSPStartupPreflight
from homesec.sources.rtsp.url_derivation import derive_detect_rtsp_url

logger = logging.getLogger(__name__)

RTSP_TEST_CONNECTION_TIMEOUT_S = 10.0
RTSP_PREFLIGHT_COMMAND_TIMEOUT_CAP_S = 8.0
ProbeResultT = TypeVar("ProbeResultT")


async def _run_owned_inspection(
    inspector: InspectionGroup, work: Callable[[], ProbeResultT], *, timeout_s: float
) -> ProbeResultT:
    """Wait for the worker's concrete cleanup before releasing request ownership."""

    def run() -> ProbeResultT:
        with inspector:
            return work()

    worker = asyncio.create_task(asyncio.to_thread(run))
    try:
        return await asyncio.wait_for(asyncio.shield(worker), timeout=timeout_s)
    except (asyncio.CancelledError, TimeoutError):

        async def cleanup() -> None:
            cleanup_error: Exception | None = None
            try:
                await asyncio.to_thread(inspector.cancel)
            except Exception as exc:
                cleanup_error = exc
            finally:
                try:
                    await worker
                except Exception:
                    # Cancellation supersedes the operation, but never its ownership.
                    pass
            if cleanup_error is not None:
                raise cleanup_error

        cleanup_task = asyncio.create_task(cleanup())
        # A second cancellation (for example the registry's outer timeout) must not
        # detach a camera-owning thread or let TemporaryDirectory remove a live writer.
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                continue
        cleanup_task.result()
        raise


async def rtsp_inspection_readiness(
    *, context: SetupProbeContext, timeout_s: float
) -> PreflightCheckResponse:
    """Inspect the configured native build without opening any camera input."""
    start = time.perf_counter()
    inspector = InspectionGroup(
        helper_path=select_helper_path(context.preview), total_timeout_s=timeout_s
    )
    try:
        evidence = await _run_owned_inspection(
            inspector,
            lambda: inspector.request("inspect_capabilities", timeout_s=timeout_s),
            timeout_s=timeout_s,
        )
        if not isinstance(evidence, CapabilityEvidence) or not evidence.baseline_ready():
            raise InspectionFailure("local_capability_unavailable")
        message = (
            f"Native inspection ready: FFmpeg {evidence.library_version}, build {evidence.build_id}. "
            "Recording and preview execution dependencies remain separate."
        )
        passed = True
    except TimeoutError:
        message, passed = "Native inspection readiness timed out", False
    except InspectionFailure as exc:
        message, passed = str(exc), False
    except Exception:
        message, passed = "Native inspection helper unavailable or incompatible", False
    return PreflightCheckResponse(
        name="ffmpeg",
        passed=passed,
        message=message,
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )


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


@setup_probe(
    "camera",
    "rtsp",
    timeout_s=RTSP_TEST_CONNECTION_TIMEOUT_S + 1.0,
    contextual=True,
    readiness=rtsp_inspection_readiness,
)
async def test_rtsp_camera_connection(
    *,
    config: dict[str, object],
    context: SetupProbeContext | None = None,
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
        inspector = InspectionGroup(
            helper_path=select_helper_path(context.preview if context is not None else None),
            total_timeout_s=RTSP_TEST_CONNECTION_TIMEOUT_S,
            connect_timeout_s=float(validated.stream.connect_timeout_s),
            io_timeout_s=float(validated.stream.io_timeout_s),
        )
        preflight = RTSPStartupPreflight(
            output_dir=Path(temp_dir),
            rtsp_connect_timeout_s=float(validated.stream.connect_timeout_s),
            rtsp_io_timeout_s=float(validated.stream.io_timeout_s),
            command_timeout_s=min(
                RTSP_TEST_CONNECTION_TIMEOUT_S,
                RTSP_PREFLIGHT_COMMAND_TIMEOUT_CAP_S,
            ),
            inspection=inspector,
        )
        try:
            outcome = await _run_owned_inspection(
                inspector,
                lambda: preflight.run(
                    camera_name=validated.camera_name or SETUP_TEST_CAMERA_NAME,
                    primary_rtsp_url=primary_url,
                    detect_rtsp_url=detect_url,
                ),
                timeout_s=RTSP_TEST_CONNECTION_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return build_test_connection_response(
                success=False,
                message=f"RTSP probe timed out after {RTSP_TEST_CONNECTION_TIMEOUT_S:.1f}s.",
                start=start,
            )
        except InspectionFailure as exc:
            return build_test_connection_response(
                success=False,
                message=str(exc),
                start=start,
                details={
                    "stage": "inspection",
                    "camera_key": build_camera_key(
                        validated.camera_name or SETUP_TEST_CAMERA_NAME, primary_url
                    ),
                },
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
