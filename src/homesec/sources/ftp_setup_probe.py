"""Setup-only FTP source connectivity probe."""

from __future__ import annotations

import asyncio
import logging
import socket
import time

from pydantic import ValidationError

from homesec.models.setup import TestConnectionResponse
from homesec.plugins.registry import PluginType, validate_plugin
from homesec.services.setup_probe_support import (
    SETUP_TEST_CAMERA_NAME,
    build_test_connection_response,
    format_validation_error,
)
from homesec.services.setup_probes import setup_probe
from homesec.sources.ftp import FtpSourceConfig

logger = logging.getLogger(__name__)

FTP_TEST_CONNECTION_TIMEOUT_S = 5.0


def probe_tcp_bind(host: str, port: int) -> int:
    """Bind a temporary TCP socket and return the bound port."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return int(sock.getsockname()[1])


@setup_probe("camera", "ftp", timeout_s=FTP_TEST_CONNECTION_TIMEOUT_S + 1.0)
async def test_ftp_camera_connection(
    *,
    config: dict[str, object],
) -> TestConnectionResponse:
    """Validate FTP source config and confirm the listen socket can bind."""
    start = time.perf_counter()
    try:
        validated = validate_plugin(
            PluginType.SOURCE,
            "ftp",
            config,
            camera_name=SETUP_TEST_CAMERA_NAME,
        )
    except ValidationError as exc:
        return build_test_connection_response(
            success=False,
            message=format_validation_error(exc),
            start=start,
        )

    if not isinstance(validated, FtpSourceConfig):
        return build_test_connection_response(
            success=False,
            message=f"Unexpected ftp config model: {type(validated).__name__}",
            start=start,
        )

    try:
        bound_port = await asyncio.wait_for(
            asyncio.to_thread(probe_tcp_bind, validated.host, int(validated.port)),
            timeout=FTP_TEST_CONNECTION_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        return build_test_connection_response(
            success=False,
            message=f"FTP bind probe timed out after {FTP_TEST_CONNECTION_TIMEOUT_S:.1f}s.",
            start=start,
        )
    except OSError:
        logger.warning("FTP setup test probe failed", exc_info=True)
        return build_test_connection_response(
            success=False,
            message="FTP bind probe failed. Check bind host/port availability and permissions.",
            start=start,
        )

    return build_test_connection_response(
        success=True,
        message="FTP listen address is available.",
        start=start,
        details={"host": validated.host, "bound_port": bound_port},
    )
