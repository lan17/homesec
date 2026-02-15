"""Tests for OpenAPI export helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from homesec.api import openapi_export
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


def test_openapi_export_parser_requires_output() -> None:
    # Given: The OpenAPI export CLI parser
    parser = openapi_export._build_parser()

    # When: Parsing arguments without the required --output flag
    with pytest.raises(SystemExit):
        parser.parse_args([])

    # Then: argparse exits due to missing required argument


def test_openapi_export_main_writes_requested_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given: A target output path and CLI argv containing --output
    output_path = tmp_path / "cli-openapi.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "openapi_export",
            "--output",
            str(output_path),
        ],
    )

    # When: Running OpenAPI export CLI entrypoint
    openapi_export.main()

    # Then: output file is created with expected API paths
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "/api/v1/health" in payload["paths"]
