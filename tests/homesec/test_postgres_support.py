"""Tests for Postgres helper utilities."""

from __future__ import annotations

import pytest

from homesec.postgres_support import (
    TEST_DB_SCHEMA_ENABLE_ENV,
    TEST_DB_SCHEMA_ENV,
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


@pytest.mark.parametrize("invalid_schema", ["Uppercase", "1schema", "has-dash"])
def test_resolve_test_db_schema_rejects_invalid_values(invalid_schema: str) -> None:
    """resolve_test_db_schema should validate non-empty env values."""
    # Given: An environment mapping with an invalid schema override
    env = {TEST_DB_SCHEMA_ENV: invalid_schema}

    # When/Then: Resolving the schema fails with ValueError
    with pytest.raises(ValueError, match="Invalid Postgres schema"):
        resolve_test_db_schema(env)


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


@pytest.mark.parametrize("enabled_value", ["1", "true", "TRUE", "yes", "on"])
def test_test_db_schema_enabled_accepts_truthy_values(enabled_value: str) -> None:
    """is_test_db_schema_enabled should accept the documented truthy enable values."""
    # Given: An environment mapping with an explicit enable flag
    env = {
        TEST_DB_SCHEMA_ENV: "hs_pytest_abc12345",
        TEST_DB_SCHEMA_ENABLE_ENV: enabled_value,
    }

    # When: Evaluating whether test schema scoping is enabled
    enabled = is_test_db_schema_enabled(env)

    # Then: The harness opt-in is recognized as enabled
    assert enabled is True


@pytest.mark.parametrize("disabled_value", ["", "0", "false", "FALSE", "no", "off"])
def test_test_db_schema_enabled_rejects_falsy_values(disabled_value: str) -> None:
    """is_test_db_schema_enabled should stay off for missing or falsy values."""
    # Given: An environment mapping without a truthy enable flag
    env = {
        TEST_DB_SCHEMA_ENV: "hs_pytest_abc12345",
        TEST_DB_SCHEMA_ENABLE_ENV: disabled_value,
    }

    # When: Evaluating whether test schema scoping is enabled
    enabled = is_test_db_schema_enabled(env)

    # Then: Schema scoping remains disabled
    assert enabled is False


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
