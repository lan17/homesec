"""Setup-only ONVIF connectivity probe."""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field, ValidationError

from homesec.models.setup import TestConnectionResponse
from homesec.onvif.client import OnvifCameraClient
from homesec.onvif.discovery import discover_cameras
from homesec.onvif.service import (
    DEFAULT_ONVIF_PORT,
    OnvifProbeError,
    OnvifProbeOptions,
    OnvifProbeTimeoutError,
    OnvifService,
)
from homesec.services.setup_probe_support import (
    build_test_connection_response,
    format_validation_error,
)
from homesec.services.setup_probes import setup_probe

logger = logging.getLogger(__name__)

ONVIF_TEST_CONNECTION_TIMEOUT_CAP_S = 30.0
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


@setup_probe("camera", "onvif", timeout_s=ONVIF_TEST_CONNECTION_TIMEOUT_CAP_S + 1.0)
async def test_onvif_camera_connection(
    *,
    config: dict[str, object],
) -> TestConnectionResponse:
    """Validate ONVIF probe settings and fetch stream/profile metadata."""
    start = time.perf_counter()
    try:
        request = OnvifSetupProbeConfig.model_validate(config)
    except ValidationError as exc:
        return build_test_connection_response(
            success=False,
            message=format_validation_error(exc),
            start=start,
        )

    service = OnvifService(
        discover_fn=discover_cameras,
        client_factory=OnvifCameraClient,
    )
    timeout_s = request.timeout_s
    try:
        probe_result = await service.probe(
            OnvifProbeOptions(
                host=request.host,
                username=request.username,
                password=request.password,
                port=int(request.port),
                timeout_s=timeout_s,
            )
        )
    except OnvifProbeTimeoutError:
        return build_test_connection_response(
            success=False,
            message=f"ONVIF probe timed out after {timeout_s:.1f}s.",
            start=start,
        )
    except OnvifProbeError:
        logger.warning("ONVIF setup test probe failed", exc_info=True)
        return build_test_connection_response(
            success=False,
            message="ONVIF probe failed. Check host, credentials, and camera reachability.",
            start=start,
        )

    return build_test_connection_response(
        success=True,
        message="ONVIF probe succeeded.",
        start=start,
        details={
            "profiles": len(probe_result.profiles),
            "streams": len(probe_result.streams),
        },
    )
