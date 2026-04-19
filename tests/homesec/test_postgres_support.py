"""Tests for Postgres helper utilities."""

from __future__ import annotations

import pytest

from homesec.postgres_support import (
    build_async_engine_kwargs,
    is_test_db_schema_enabled,
    resolve_test_db_schema,
    validate_schema_name,
)


def test_validate_schema_name_accepts_lowercase_max_length() -> None:
    """validate_schema_name should accept valid lowercase identifiers up to 63 chars."""
    # Given: A valid 63-character lowercase schema name
    schema = "a" * 63

    # When: Validating the schema name
    validated = validate_schema_name(schema)

    # Then: The original schema name is returned unchanged
    assert validated == schema


@pytest.mark.parametrize(
    "invalid_schema",
    [
        "",
        "1schema",
        "has-dash",
        'has"quote',
        "Uppercase",
        "a" * 64,
    ],
)
def test_validate_schema_name_rejects_invalid_values(invalid_schema: str) -> None:
    """validate_schema_name should reject invalid test schema identifiers."""
    # Given: An invalid schema identifier candidate

    # When/Then: Validation fails with ValueError
    with pytest.raises(ValueError, match="Invalid Postgres schema"):
        validate_schema_name(invalid_schema)


def test_resolve_test_db_schema_returns_none_for_empty_string() -> None:
    """resolve_test_db_schema should treat an empty env var as unset."""
    # Given: An environment mapping with an empty test schema override
    env = {"HOMESEC_TEST_DB_SCHEMA": ""}

    # When: Resolving the configured test schema
    schema = resolve_test_db_schema(env)

    # Then: No schema override is returned
    assert schema is None


def test_build_async_engine_kwargs_ignores_test_env_without_explicit_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_async_engine_kwargs should ignore the test env unless schema is passed explicitly."""
    # Given: A test-schema env var that should remain test-harness-only
    monkeypatch.setenv("HOMESEC_TEST_DB_SCHEMA", "hs_pytest_abc12345")

    # When: Building engine kwargs without an explicit schema
    kwargs = build_async_engine_kwargs()

    # Then: No search_path override is added implicitly
    assert kwargs == {}


def test_test_db_schema_enabled_requires_explicit_opt_in() -> None:
    """is_test_db_schema_enabled should only turn on when the harness opts in."""
    # Given: Environments with and without the explicit enable flag
    disabled_env = {"HOMESEC_TEST_DB_SCHEMA": "hs_pytest_abc12345"}
    enabled_env = {
        "HOMESEC_TEST_DB_SCHEMA": "hs_pytest_abc12345",
        "HOMESEC_ENABLE_TEST_DB_SCHEMA": "1",
    }

    # When: Evaluating whether test schema scoping is enabled
    disabled = is_test_db_schema_enabled(disabled_env)
    enabled = is_test_db_schema_enabled(enabled_env)

    # Then: Schema scoping stays off unless the harness explicitly enables it
    assert disabled is False
    assert enabled is True


def test_build_async_engine_kwargs_sets_search_path_for_schema() -> None:
    """build_async_engine_kwargs should set search_path for the requested schema."""
    # Given: An explicit schema with no caller-specified search_path

    # When: Building engine kwargs
    kwargs = build_async_engine_kwargs(schema="hs_pytest_abc12345")

    # Then: search_path is set to the requested schema
    assert kwargs == {"connect_args": {"server_settings": {"search_path": "hs_pytest_abc12345"}}}


def test_build_async_engine_kwargs_rejects_conflicting_search_path() -> None:
    """build_async_engine_kwargs should reject conflicting caller-provided search_path values."""
    # Given: An explicit schema and conflicting caller-provided server settings
    engine_kwargs = {"connect_args": {"server_settings": {"search_path": "other_schema"}}}

    # When/Then: Building engine kwargs fails loudly instead of silently ignoring schema
    with pytest.raises(ValueError, match="search_path conflicts with schema"):
        build_async_engine_kwargs(schema="hs_pytest_abc12345", engine_kwargs=engine_kwargs)
