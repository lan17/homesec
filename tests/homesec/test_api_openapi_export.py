"""Tests for OpenAPI export helpers."""

from __future__ import annotations

import json
from pathlib import Path

from homesec.api.openapi_export import build_openapi_schema, write_openapi_schema


def test_build_openapi_schema_includes_health_route() -> None:
    # Given: API contract routes are registered for schema generation

    # When: Building the OpenAPI schema in memory
    schema = build_openapi_schema()

    # Then: Versioned health route and response schema should be present
    assert "/api/v1/health" in schema["paths"]
    assert "HealthResponse" in schema["components"]["schemas"]


def test_write_openapi_schema_writes_deterministic_json(tmp_path: Path) -> None:
    # Given: A target output path for exported schema
    output_path = tmp_path / "openapi.json"

    # When: Writing schema to disk
    write_openapi_schema(output_path)

    # Then: JSON should be valid, newline-terminated, and include key contract paths
    text = output_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    payload = json.loads(text)
    assert "/api/v1/health" in payload["paths"]
