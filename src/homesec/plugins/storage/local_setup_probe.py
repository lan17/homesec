"""Setup-only local storage connectivity probe."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from pydantic import ValidationError

from homesec.models.setup import TestConnectionResponse
from homesec.plugins.registry import PluginType, validate_plugin
from homesec.plugins.storage.local import LocalStorageConfig
from homesec.services.setup_probe_support import (
    build_test_connection_response,
    format_validation_error,
    nearest_existing_parent,
)
from homesec.services.setup_probes import setup_probe


@setup_probe("storage", "local")
async def test_local_storage_connection(*, config: dict[str, object]) -> TestConnectionResponse:
    """Validate local storage config and confirm the target path is usable."""
    start = time.perf_counter()
    try:
        validated = validate_plugin(PluginType.STORAGE, "local", config)
    except ValidationError as exc:
        return build_test_connection_response(
            success=False,
            message=format_validation_error(exc),
            start=start,
        )

    if not isinstance(validated, LocalStorageConfig):
        return build_test_connection_response(
            success=False,
            message=f"Unexpected local storage config model: {type(validated).__name__}",
            start=start,
        )

    root = Path(validated.root).expanduser()
    if await asyncio.to_thread(root.exists):
        if not await asyncio.to_thread(root.is_dir):
            return build_test_connection_response(
                success=False,
                message=f"Storage root is not a directory: {root}",
                start=start,
            )
        if not await asyncio.to_thread(os.access, root, os.R_OK | os.W_OK | os.X_OK):
            return build_test_connection_response(
                success=False,
                message=f"Storage root is not readable/writable: {root}",
                start=start,
            )
        resolved_root = await asyncio.to_thread(root.resolve)
        return build_test_connection_response(
            success=True,
            message="Local storage root is accessible.",
            start=start,
            details={"root": str(resolved_root)},
        )

    parent = await asyncio.to_thread(nearest_existing_parent, root)
    if parent is None:
        return build_test_connection_response(
            success=False,
            message=f"No existing parent directory for storage root: {root}",
            start=start,
        )
    if not await asyncio.to_thread(os.access, parent, os.W_OK | os.X_OK):
        return build_test_connection_response(
            success=False,
            message=f"Storage root parent is not writable: {parent}",
            start=start,
        )
    return build_test_connection_response(
        success=True,
        message="Local storage root can be created.",
        start=start,
        details={"root": str(root), "writable_parent": str(parent)},
    )
