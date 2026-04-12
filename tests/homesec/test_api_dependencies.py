"""Tests for API dependency helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from starlette.requests import Request

from homesec.api.dependencies import require_database, require_normal_mode, verify_api_key
from homesec.api.errors import APIError
from homesec.models.config import FastAPIServerConfig


class _StubRepository:
    def __init__(self, *, ok: bool) -> None:
        self._ok = ok
        self.ping_calls = 0

    async def ping(self) -> bool:
        self.ping_calls += 1
        return self._ok


@dataclass
class _MinimalDatabaseApp:
    bootstrap_mode: bool
    repository: _StubRepository


@dataclass
class _MinimalModeApp:
    bootstrap_mode: bool


@dataclass
class _MinimalAuthApp:
    server_config: FastAPIServerConfig


def _request_with_bearer(token: str) -> Request:
    return Request(
        {
            "type": "http",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
        }
    )


@pytest.mark.asyncio
async def test_require_database_accepts_minimal_protocol_app() -> None:
    """require_database should only need bootstrap flag plus repository ping."""
    # Given: A minimal app-shaped object with bootstrap flag and repository
    repository = _StubRepository(ok=True)
    app = _MinimalDatabaseApp(bootstrap_mode=False, repository=repository)

    # When: Enforcing database availability
    await require_database(app)

    # Then: The repository ping succeeds without needing a full Application
    assert repository.ping_calls == 1


@pytest.mark.asyncio
async def test_require_normal_mode_accepts_minimal_protocol_app() -> None:
    """require_normal_mode should only need the bootstrap flag."""
    # Given: A minimal app-shaped object in normal mode
    app = _MinimalModeApp(bootstrap_mode=False)

    # When: Enforcing normal runtime mode
    await require_normal_mode(app)

    # Then: No error is raised without any wider Application surface


@pytest.mark.asyncio
async def test_verify_api_key_accepts_minimal_auth_protocol_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """verify_api_key should only require server_config plus request headers."""
    # Given: A minimal app-shaped object exposing only server_config
    monkeypatch.setenv("HOMESEC_API_KEY", "secret-key")
    app = _MinimalAuthApp(
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY")
    )
    request = _request_with_bearer("secret-key")

    # When: Verifying the request API key
    await verify_api_key(request, app)

    # Then: Auth succeeds without needing a full Application


@pytest.mark.asyncio
async def test_verify_api_key_rejects_invalid_token_with_minimal_auth_protocol_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """verify_api_key should still reject invalid bearer tokens."""
    # Given: A minimal auth app with a configured API key and wrong bearer token
    monkeypatch.setenv("HOMESEC_API_KEY", "secret-key")
    app = _MinimalAuthApp(
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY")
    )
    request = _request_with_bearer("wrong-key")

    # When: Verifying the request API key
    with pytest.raises(APIError) as exc_info:
        await verify_api_key(request, app)

    # Then: The helper returns the canonical unauthorized API error
    assert exc_info.value.status_code == 401
    assert exc_info.value.error_code == "UNAUTHORIZED"
