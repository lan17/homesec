"""Tests for setup-only probe registry helpers."""

from __future__ import annotations

import pytest

from homesec.models.setup import TestConnectionResponse as SetupTestConnectionResponse
from homesec.services.setup_probes import SetupProbeRegistry


def test_setup_probe_registry_registers_and_looks_up_probe() -> None:
    """Registry should register probes by target/backend and normalize lookup keys."""
    # Given: A fresh setup probe registry and a probe handler for one backend
    registry = SetupProbeRegistry()

    async def _probe(*, config: dict[str, object]) -> SetupTestConnectionResponse:
        _ = config
        return SetupTestConnectionResponse(success=True, message="ok")

    registry.register("camera", "onvif")(_probe)

    # When: Looking up the probe with mixed-case backend input
    probe = registry.get("camera", " ONVIF ")

    # Then: Registry returns the registered probe and backend listing is normalized
    assert probe is _probe
    assert registry.get_backends("camera") == ["onvif"]


def test_setup_probe_registry_rejects_duplicate_registration() -> None:
    """Registry should reject duplicate target/backend registrations."""
    # Given: A registry with one already-registered probe
    registry = SetupProbeRegistry()

    async def _probe_one(*, config: dict[str, object]) -> SetupTestConnectionResponse:
        _ = config
        return SetupTestConnectionResponse(success=True, message="one")

    async def _probe_two(*, config: dict[str, object]) -> SetupTestConnectionResponse:
        _ = config
        return SetupTestConnectionResponse(success=True, message="two")

    registry.register("storage", "local")(_probe_one)

    # When / Then: Registering the same target/backend again raises a clear error
    with pytest.raises(ValueError, match="already registered"):
        registry.register("storage", "local")(_probe_two)


def test_setup_probe_registry_tracks_custom_timeout_budget() -> None:
    """Registry should preserve explicit timeout metadata for registered probes."""
    # Given: A probe registered with a non-default timeout budget
    registry = SetupProbeRegistry()

    async def _probe(*, config: dict[str, object]) -> SetupTestConnectionResponse:
        _ = config
        return SetupTestConnectionResponse(success=True, message="ok")

    registry.register("camera", "rtsp", timeout_s=11.0)(_probe)

    # When / Then: Lookup returns the probe and its configured timeout budget
    assert registry.get("camera", "rtsp") is _probe
    assert registry.get_timeout("camera", "rtsp") == 11.0
