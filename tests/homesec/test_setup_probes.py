"""Tests for setup-only probe registry helpers."""

from __future__ import annotations

import pytest

from homesec.models.setup import TestConnectionResponse as SetupTestConnectionResponse
from homesec.services import setup_probes
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


def test_load_builtin_setup_probes_registers_backend_adjacent_modules() -> None:
    """Builtin loader should expose built-in camera and storage probe backends."""
    # Given / When: Loading the builtin setup probes through the shared loader
    setup_probes.load_builtin_setup_probes()

    # Then: The registry reports backend-adjacent builtins without setup.py-owned decorators
    assert setup_probes.get_setup_probe_backends("camera") == [
        "ftp",
        "local_folder",
        "onvif",
        "rtsp",
    ]
    assert setup_probes.get_setup_probe_backends("storage") == ["local"]
