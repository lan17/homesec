"""Setup-only local-folder source connectivity probe."""

from __future__ import annotations

import asyncio
import os
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
from homesec.sources.local_folder import LocalFolderSourceConfig


@setup_probe("camera", "local_folder")
async def test_local_folder_camera_connection(
    *,
    config: dict[str, object],
) -> TestConnectionResponse:
    """Validate local-folder config and ensure the watch directory is accessible."""
    start = time.perf_counter()
    try:
        validated = validate_plugin(
            PluginType.SOURCE,
            "local_folder",
            config,
            camera_name=SETUP_TEST_CAMERA_NAME,
        )
    except ValidationError as exc:
        return build_test_connection_response(
            success=False,
            message=format_validation_error(exc),
            start=start,
        )

    if not isinstance(validated, LocalFolderSourceConfig):
        return build_test_connection_response(
            success=False,
            message=f"Unexpected local_folder config model: {type(validated).__name__}",
            start=start,
        )

    watch_dir = Path(validated.watch_dir).expanduser()
    if not await asyncio.to_thread(watch_dir.exists):
        return build_test_connection_response(
            success=False,
            message=f"Watch directory does not exist: {watch_dir}",
            start=start,
        )
    if not await asyncio.to_thread(watch_dir.is_dir):
        return build_test_connection_response(
            success=False,
            message=f"Watch path is not a directory: {watch_dir}",
            start=start,
        )
    if not await asyncio.to_thread(os.access, watch_dir, os.R_OK | os.W_OK | os.X_OK):
        return build_test_connection_response(
            success=False,
            message=f"Watch directory is not readable/writable: {watch_dir}",
            start=start,
        )

    resolved_watch_dir = await asyncio.to_thread(watch_dir.resolve)
    return build_test_connection_response(
        success=True,
        message="Local folder path is accessible.",
        start=start,
        details={"watch_dir": str(resolved_watch_dir)},
    )
