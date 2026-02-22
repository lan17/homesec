"""WS-Discovery helpers for finding ONVIF devices on a local network."""

from __future__ import annotations

import inspect
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

try:
    from wsdiscovery.discovery import (  # type: ignore[import-untyped]
        ThreadedWSDiscovery as _ThreadedWSDiscovery,
    )
except Exception:  # pragma: no cover - exercised via dependency guard tests
    _ThreadedWSDiscovery = None


@dataclass(frozen=True, slots=True)
class DiscoveredCamera:
    """Discovered ONVIF endpoint metadata."""

    ip: str
    xaddr: str
    scopes: tuple[str, ...]
    types: tuple[str, ...]


def discover_cameras(timeout_s: float = 8.0, *, attempts: int = 2) -> list[DiscoveredCamera]:
    """Run WS-Discovery and return discovered ONVIF cameras."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    discovery_class = _require_wsdiscovery_class()
    with _suppress_wsdiscovery_interface_warnings():
        discovery = discovery_class()
        discovery.start()

        try:
            services: list[Any] = []
            for attempt in range(attempts):
                services.extend(_search_services(discovery, timeout_s=timeout_s))
                if _parse_discovery_services(services):
                    break
                if attempt < (attempts - 1):
                    time.sleep(0.2)
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


def _search_services(discovery: Any, *, timeout_s: float) -> list[Any]:
    search_services = discovery.searchServices

    try:
        signature = inspect.signature(search_services)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        if "timeout_s" in signature.parameters:
            return list(search_services(timeout_s=timeout_s))
        if "timeout" in signature.parameters:
            return list(search_services(timeout=timeout_s))

    return list(search_services())


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
    wsdiscovery_logger = logging.getLogger("threading")
    original_level = wsdiscovery_logger.level
    wsdiscovery_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        wsdiscovery_logger.setLevel(original_level)


__all__ = ["DiscoveredCamera", "discover_cameras"]
