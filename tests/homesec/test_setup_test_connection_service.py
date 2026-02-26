"""Tests for setup test-connection service flow."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from homesec.models.setup import TestConnectionRequest as SetupTestConnectionRequest
from homesec.services import setup as setup_service


@dataclass
class _StubApp:
    """Minimal app stub for setup service API."""


class _DummyConfig(BaseModel):
    value: str = "ok"


class _StubPingPlugin:
    def __init__(self, *, ping_result: bool = True, delay_s: float = 0.0) -> None:
        self._ping_result = ping_result
        self._delay_s = delay_s
        self.shutdown_calls = 0

    async def ping(self) -> bool:
        if self._delay_s > 0:
            await asyncio.sleep(self._delay_s)
        return self._ping_result

    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout
        self.shutdown_calls += 1


@pytest.mark.asyncio
async def test_test_connection_camera_local_folder_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Camera local_folder test should pass when watch dir is accessible."""
    # Given: A valid local_folder config that points at an existing writable directory
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir(parents=True)

    def _fake_validate_plugin(
        plugin_type: object,
        backend: str,
        config: object,
        **runtime_context: object,
    ) -> object:
        _ = (plugin_type, config, runtime_context)
        assert backend == "local_folder"
        return setup_service.LocalFolderSourceConfig(watch_dir=str(watch_dir))

    monkeypatch.setattr(setup_service, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["local_folder"]
        if plugin_type == setup_service.PluginType.SOURCE
        else [],
    )
    request = SetupTestConnectionRequest(type="camera", backend="local_folder", config={})

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: The connection test succeeds and reports the checked directory
    assert response.success is True
    assert "accessible" in response.message
    assert response.details is not None
    assert response.details["watch_dir"] == str(watch_dir.resolve())


@pytest.mark.asyncio
async def test_test_connection_camera_local_folder_missing_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Camera local_folder test should fail when watch dir is missing."""
    # Given: A local_folder config that points at a missing directory
    watch_dir = tmp_path / "missing-watch-dir"

    def _fake_validate_plugin(
        plugin_type: object,
        backend: str,
        config: object,
        **runtime_context: object,
    ) -> object:
        _ = (plugin_type, config, runtime_context)
        assert backend == "local_folder"
        return setup_service.LocalFolderSourceConfig(watch_dir=str(watch_dir))

    monkeypatch.setattr(setup_service, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["local_folder"]
        if plugin_type == setup_service.PluginType.SOURCE
        else [],
    )
    request = SetupTestConnectionRequest(type="camera", backend="local_folder", config={})

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: The connection test reports a non-successful validation outcome
    assert response.success is False
    assert "does not exist" in response.message


@pytest.mark.asyncio
async def test_test_connection_notifier_uses_ping_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Notifier test should ping ephemeral plugin and always shut it down."""
    # Given: A notifier backend that validates and creates a pingable ephemeral plugin
    plugin = _StubPingPlugin(ping_result=True)

    def _fake_get_plugin_names(plugin_type: setup_service.PluginType) -> list[str]:
        assert plugin_type == setup_service.PluginType.NOTIFIER
        return ["mqtt"]

    def _fake_validate_plugin(
        plugin_type: setup_service.PluginType,
        backend: str,
        config: dict[str, Any] | BaseModel,
        **runtime_context: object,
    ) -> BaseModel:
        _ = (config, runtime_context)
        assert plugin_type == setup_service.PluginType.NOTIFIER
        assert backend == "mqtt"
        return _DummyConfig()

    def _fake_load_plugin(
        plugin_type: setup_service.PluginType,
        backend: str,
        config: dict[str, Any] | BaseModel,
        **runtime_context: object,
    ) -> _StubPingPlugin:
        _ = (config, runtime_context)
        assert plugin_type == setup_service.PluginType.NOTIFIER
        assert backend == "mqtt"
        return plugin

    monkeypatch.setattr(setup_service, "get_plugin_names", _fake_get_plugin_names)
    monkeypatch.setattr(setup_service, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(setup_service, "load_plugin", _fake_load_plugin)
    request = SetupTestConnectionRequest(type="notifier", backend="mqtt", config={})

    # When: Running notifier test-connection
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Probe succeeds and plugin shutdown executes exactly once
    assert response.success is True
    assert plugin.shutdown_calls == 1


@pytest.mark.asyncio
async def test_test_connection_notifier_timeout_returns_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Notifier test should return success=false when ping exceeds timeout."""
    # Given: A notifier plugin whose ping blocks longer than timeout budget
    plugin = _StubPingPlugin(ping_result=True, delay_s=0.05)

    def _fake_get_plugin_names(plugin_type: setup_service.PluginType) -> list[str]:
        assert plugin_type == setup_service.PluginType.NOTIFIER
        return ["mqtt"]

    def _fake_validate_plugin(
        plugin_type: setup_service.PluginType,
        backend: str,
        config: dict[str, Any] | BaseModel,
        **runtime_context: object,
    ) -> BaseModel:
        _ = (config, runtime_context)
        assert plugin_type == setup_service.PluginType.NOTIFIER
        assert backend == "mqtt"
        return _DummyConfig()

    def _fake_load_plugin(
        plugin_type: setup_service.PluginType,
        backend: str,
        config: dict[str, Any] | BaseModel,
        **runtime_context: object,
    ) -> _StubPingPlugin:
        _ = (config, runtime_context)
        assert plugin_type == setup_service.PluginType.NOTIFIER
        assert backend == "mqtt"
        return plugin

    monkeypatch.setattr(setup_service, "_TEST_CONNECTION_TIMEOUT_S", 0.01)
    monkeypatch.setattr(setup_service, "get_plugin_names", _fake_get_plugin_names)
    monkeypatch.setattr(setup_service, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(setup_service, "load_plugin", _fake_load_plugin)
    request = SetupTestConnectionRequest(type="notifier", backend="mqtt", config={})

    # When: Running notifier test-connection
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Probe reports timeout as a non-successful test outcome
    assert response.success is False
    assert "timed out" in response.message
    assert plugin.shutdown_calls == 1


@pytest.mark.asyncio
async def test_test_connection_raises_for_unknown_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown backend should raise typed request error for route-level 400 mapping."""

    # Given: A storage test request for a backend that is not registered
    def _fake_get_plugin_names(plugin_type: setup_service.PluginType) -> list[str]:
        assert plugin_type == setup_service.PluginType.STORAGE
        return ["local", "dropbox"]

    monkeypatch.setattr(setup_service, "get_plugin_names", _fake_get_plugin_names)
    request = SetupTestConnectionRequest(type="storage", backend="s3", config={})

    # When / Then: Dispatch raises a typed request error with available backends attached
    with pytest.raises(setup_service.SetupTestConnectionRequestError) as exc_info:
        await setup_service.test_connection(request, _StubApp())

    assert "Unknown storage backend" in str(exc_info.value)
    assert exc_info.value.available_backends == ["local", "dropbox"]
