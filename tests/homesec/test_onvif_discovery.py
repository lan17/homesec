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


def test_discover_cameras_probes_each_type_separately_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """discover_cameras should probe Device and NVT types separately and merge results."""

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

        def __init__(self, **kwargs: Any) -> None:
            self.init_kwargs = kwargs
            self.started = False
            self.stopped = False
            self.search_calls: list[dict[str, Any]] = []
            self.__class__.instances.append(self)

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

        def searchServices(self, **kwargs: Any) -> list[_FakeService]:
            self.search_calls.append(kwargs)
            # Return overlapping results to exercise dedup across probes.
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

    # Given: WS-Discovery returns duplicated XAddr entries across multiple probes
    _FakeDiscovery.instances = []
    monkeypatch.setattr("homesec.onvif.discovery._ThreadedWSDiscovery", _FakeDiscovery)

    # When: Running discovery with 1 attempt
    cameras = discover_cameras(attempts=1)

    # Then: Results are deduplicated across both type probes
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

    inst = _FakeDiscovery.instances[0]
    assert inst.started is True
    assert inst.stopped is True

    # Constructor should receive relates_to and ttl
    assert inst.init_kwargs["relates_to"] is True
    assert inst.init_kwargs["ttl"] == 4

    # searchServices should be called twice (once per ONVIF type) even with
    # attempts=1, because each type requires its own probe.
    assert len(inst.search_calls) == 2


def test_discover_cameras_runs_all_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """discover_cameras should run every attempt round, not stop early."""

    class _FakeDiscovery:
        instances: list[Any] = []

        def __init__(self, **kwargs: Any) -> None:
            self.search_count = 0
            self.__class__.instances.append(self)

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def searchServices(self, **kwargs: Any) -> list[Any]:
            self.search_count += 1
            return []

    _FakeDiscovery.instances = []
    monkeypatch.setattr("homesec.onvif.discovery._ThreadedWSDiscovery", _FakeDiscovery)

    # When: Running with 3 attempts
    discover_cameras(attempts=3)

    # Then: All 3 rounds run, each probing 2 types = 6 total calls
    assert _FakeDiscovery.instances[0].search_count == 6


def test_discover_cameras_stops_discovery_when_search_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """discover_cameras should always stop discovery even when search raises."""

    class _FailingDiscovery:
        instances: list[Any] = []

        def __init__(self, **kwargs: Any) -> None:
            self.stopped = False
            self.__class__.instances.append(self)

        def start(self) -> None:
            return None

        def stop(self) -> None:
            self.stopped = True

        def searchServices(self, **kwargs: Any) -> list[Any]:
            raise RuntimeError("network error")

    # Given: WS-Discovery search fails during scan
    _FailingDiscovery.instances = []
    monkeypatch.setattr("homesec.onvif.discovery._ThreadedWSDiscovery", _FailingDiscovery)

    # When/Then: Discovery raises and still invokes stop() for cleanup
    with pytest.raises(RuntimeError, match="network error"):
        discover_cameras()
    assert _FailingDiscovery.instances[0].stopped is True
