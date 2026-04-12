"""Shared helpers for setup-only probe implementations."""

from __future__ import annotations

import time

from pydantic import ValidationError

from homesec.models.setup import TestConnectionResponse

SETUP_TEST_CAMERA_NAME = "setup-test-camera"


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
