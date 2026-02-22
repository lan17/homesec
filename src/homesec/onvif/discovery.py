"""WS-Discovery helpers for finding ONVIF devices on a local network."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

try:
    from wsdiscovery import QName  # type: ignore[import-untyped]
    from wsdiscovery.discovery import (  # type: ignore[import-untyped]
        ThreadedWSDiscovery as _ThreadedWSDiscovery,
    )
except Exception:  # pragma: no cover - exercised via dependency guard tests
    _ThreadedWSDiscovery = None
    QName = None

# Standard ONVIF WS-Discovery type qualifiers.
# WS-Discovery Probe type matching is AND — a single probe with both types
# only returns devices advertising *both*.  We probe each type separately
# and merge results so we find devices advertising *either*.
_ONVIF_DEVICE_TYPE = ("http://www.onvif.org/ver10/device/wsdl", "Device")
_ONVIF_NVT_TYPE = ("http://www.onvif.org/ver10/network/wsdl", "NetworkVideoTransmitter")


@dataclass(frozen=True, slots=True)
class DiscoveredCamera:
    """Discovered ONVIF endpoint metadata."""

    ip: str
    xaddr: str
    scopes: tuple[str, ...]
    types: tuple[str, ...]


def discover_cameras(
    timeout_s: float = 8.0,
    *,
    attempts: int = 2,
    ttl: int = 4,
) -> list[DiscoveredCamera]:
    """Run WS-Discovery and return discovered ONVIF cameras.

    Args:
        timeout_s: Seconds to wait per probe for camera responses.
        attempts: Number of probe rounds.  All rounds always run so that
                  slow-responding cameras are not missed; results are
                  deduplicated.
        ttl: UDP multicast time-to-live.  Values >1 allow discovery across
             VLANs / routed subnets.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    discovery_class = _require_wsdiscovery_class()
    onvif_type_sets = _build_onvif_type_sets()

    with _suppress_wsdiscovery_interface_warnings():
        # relates_to=True: accept probe responses that reuse the probe's
        # MessageID — many cameras (TP-Link, Hikvision, budget Chinese
        # models) do this, and without the flag their replies are silently
        # dropped as duplicate messages.
        discovery = discovery_class(ttl=ttl, relates_to=True)
        discovery.start()

        try:
            services: list[Any] = []
            for attempt in range(attempts):
                # Probe each ONVIF type separately because WS-Discovery
                # uses AND matching — a single probe with [Device, NVT]
                # would miss cameras that only advertise one of the two.
                for type_set in onvif_type_sets:
                    services.extend(
                        _search_services(discovery, timeout_s=timeout_s, types=type_set)
                    )
                if attempt < (attempts - 1):
                    time.sleep(0.5)
            return _parse_discovery_services(services)
        finally:
            try:
                discovery.stop()
            except Exception:
                logger.warning("WS-Discovery stop() failed", exc_info=True)


def _require_wsdiscovery_class() -> type[Any]:
    if _ThreadedWSDiscovery is None:
        raise RuntimeError(
            "Missing dependency: WSDiscovery. Install with: uv pip install WSDiscovery"
        )
    return cast(type[Any], _ThreadedWSDiscovery)


def _build_onvif_type_sets() -> list[list[Any]]:
    """Return per-type probe lists for separate WS-Discovery probes.

    Each inner list is passed to a separate ``searchServices`` call so that
    the union of both device types is discovered (WS-Discovery uses AND
    matching within a single probe).
    """
    if QName is None:
        return [[]]  # single unfiltered probe as fallback
    return [
        [QName(*_ONVIF_DEVICE_TYPE)],
        [QName(*_ONVIF_NVT_TYPE)],
    ]


def _search_services(
    discovery: Any, *, timeout_s: float, types: list[Any] | None
) -> list[Any]:
    try:
        return list(discovery.searchServices(types=types, timeout=timeout_s))
    except TypeError:
        # Fallback for older WSDiscovery versions with different signatures.
        try:
            return list(discovery.searchServices(timeout=timeout_s))
        except TypeError:
            return list(discovery.searchServices())


def _parse_discovery_services(services: list[Any]) -> list[DiscoveredCamera]:
    discovered: list[DiscoveredCamera] = []
    seen: set[tuple[str, str]] = set()

    for service in services:
        xaddrs = tuple(str(value) for value in _as_iterable(_safe_call(service, "getXAddrs")))
        scopes = tuple(str(value) for value in _as_iterable(_safe_call(service, "getScopes")))
        types = tuple(str(value) for value in _as_iterable(_safe_call(service, "getTypes")))

        for xaddr in xaddrs:
            ip = _extract_ip(xaddr)
            if ip is None:
                continue

            key = (ip, xaddr)
            if key in seen:
                continue
            seen.add(key)
            discovered.append(
                DiscoveredCamera(
                    ip=ip,
                    xaddr=xaddr,
                    scopes=scopes,
                    types=types,
                )
            )

    discovered.sort(key=lambda camera: (camera.ip, camera.xaddr))
    return discovered


def _safe_call(obj: Any, method_name: str) -> Any:
    method = getattr(obj, method_name, None)
    if method is None:
        return []

    try:
        return method()
    except Exception:
        logger.debug("WS-Discovery method failed: %s", method_name, exc_info=True)
        return []


def _as_iterable(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    return (value,)


def _extract_ip(xaddr: str) -> str | None:
    parsed = urlparse(xaddr if "://" in xaddr else f"http://{xaddr}")
    if parsed.hostname:
        return parsed.hostname
    logger.debug("Could not extract host from WS-Discovery XAddr: %s", xaddr)
    return None


@contextmanager
def _suppress_wsdiscovery_interface_warnings() -> Any:
    """Suppress noisy WSDiscovery interface warnings for non-multicast interfaces."""
    wsdiscovery_logger = logging.getLogger("wsdiscovery")
    original_level = wsdiscovery_logger.level
    wsdiscovery_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        wsdiscovery_logger.setLevel(original_level)


__all__ = ["DiscoveredCamera", "discover_cameras"]
