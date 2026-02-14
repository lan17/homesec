"""OpenAPI schema export helpers for UI client generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from homesec.api.server import create_contract_app


def build_openapi_schema() -> dict[str, Any]:
    """Build the HomeSec OpenAPI schema from the registered API contract."""
    app = create_contract_app()
    schema = app.openapi()
    return schema


def write_openapi_schema(output_path: Path) -> None:
    """Write the OpenAPI schema to disk using deterministic JSON formatting."""
    schema = build_openapi_schema()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export HomeSec OpenAPI schema")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for generated OpenAPI JSON",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args()
    write_openapi_schema(args.output)


if __name__ == "__main__":
    main()
