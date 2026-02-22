"""Tests for ONVIF WS-Discovery helpers."""

from __future__ import annotations

from typing import Any

import pytest

from homesec.onvif.discovery import DiscoveredCamera, discover_cameras


def test_discover_cameras_requires_wsdiscovery_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """discover_cameras should fail with actionable message when WSDiscovery is unavailable."""
    # Given: WSDiscovery is unavailable in the runtime environment
    monkeypatch.setattr("homesec.onvif.discovery._ThreadedWSDiscovery", None)

    # When/Then: Running discovery raises a dependency error
    with pytest.raises(RuntimeError, match="Missing dependency: WSDiscovery"):
        discover_cameras()


def test_discover_cameras_rejects_invalid_attempt_count() -> None:
    """discover_cameras should validate attempt count for deterministic behavior."""
    # Given: Invalid attempt count

    # When/Then: Discovery rejects non-positive attempts
    with pytest.raises(ValueError, match="attempts must be >= 1"):
        discover_cameras(attempts=0)


def test_discover_cameras_parses_and_deduplicates_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """discover_cameras should parse service metadata and dedupe repeated XAddr records."""

    class _FakeService:
        def __init__(self, xaddrs: list[str], scopes: list[str], types: list[str]) -> None:
            self._xaddrs = xaddrs
            self._scopes = scopes
            self._types = types

        def getXAddrs(self) -> list[str]:
            return self._xaddrs

        def getScopes(self) -> list[str]:
            return self._scopes

        def getTypes(self) -> list[str]:
            return self._types

    class _FakeDiscovery:
        instances: list[Any] = []

        def __init__(self) -> None:
            self.started = False
            self.stopped = False
            self.__class__.instances.append(self)

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

        def searchServices(self) -> list[_FakeService]:
            return [
                _FakeService(
                    xaddrs=["http://192.168.1.10/onvif/device_service"],
                    scopes=["scope-a"],
                    types=["dn:NetworkVideoTransmitter"],
                ),
                _FakeService(
                    xaddrs=[
                        "http://192.168.1.10/onvif/device_service",
                        "http://192.168.1.11/onvif/device_service",
                    ],
                    scopes=["scope-b"],
                    types=["dn:NetworkVideoTransmitter"],
                ),
            ]

    # Given: WS-Discovery returns duplicated XAddr entries for one host
    monkeypatch.setattr("homesec.onvif.discovery._ThreadedWSDiscovery", _FakeDiscovery)

    # When: Running discovery
    cameras = discover_cameras()

    # Then: Results are deduplicated and normalized to unique IP + XAddr pairs
    assert cameras == [
        DiscoveredCamera(
            ip="192.168.1.10",
            xaddr="http://192.168.1.10/onvif/device_service",
            scopes=("scope-a",),
            types=("dn:NetworkVideoTransmitter",),
        ),
        DiscoveredCamera(
            ip="192.168.1.11",
            xaddr="http://192.168.1.11/onvif/device_service",
            scopes=("scope-b",),
            types=("dn:NetworkVideoTransmitter",),
        ),
    ]
    assert len(_FakeDiscovery.instances) == 1
    assert _FakeDiscovery.instances[0].started is True
    assert _FakeDiscovery.instances[0].stopped is True


def test_discover_cameras_stops_discovery_when_search_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """discover_cameras should always stop discovery even when search raises."""

    class _FailingDiscovery:
        instances: list[Any] = []

        def __init__(self) -> None:
            self.stopped = False
            self.__class__.instances.append(self)

        def start(self) -> None:
            return None

        def stop(self) -> None:
            self.stopped = True

        def searchServices(self) -> list[Any]:
            raise RuntimeError("network error")

    # Given: WS-Discovery search fails during scan
    monkeypatch.setattr("homesec.onvif.discovery._ThreadedWSDiscovery", _FailingDiscovery)

    # When/Then: Discovery raises and still invokes stop() for cleanup
    with pytest.raises(RuntimeError, match="network error"):
        discover_cameras()
    assert _FailingDiscovery.instances[0].stopped is True
