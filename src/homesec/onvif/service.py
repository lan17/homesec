"""Shared ONVIF orchestration for API and CLI flows."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from homesec.onvif.client import (
    OnvifCameraClient,
    OnvifDeviceInfo,
    OnvifMediaProfile,
    OnvifStreamUri,
)
from homesec.onvif.discovery import DiscoveredCamera, discover_cameras

logger = logging.getLogger(__name__)

DEFAULT_DISCOVER_TIMEOUT_S = 8.0
DEFAULT_DISCOVER_ATTEMPTS = 2
DEFAULT_DISCOVER_TTL = 4
DEFAULT_PROBE_TIMEOUT_S = 15.0
DEFAULT_ONVIF_PORT = 80
DEFAULT_CLIENT_CLOSE_TIMEOUT_S = 2.0


@dataclass(frozen=True, slots=True)
class OnvifDiscoverOptions:
    timeout_s: float = DEFAULT_DISCOVER_TIMEOUT_S
    attempts: int = DEFAULT_DISCOVER_ATTEMPTS
    ttl: int = DEFAULT_DISCOVER_TTL


@dataclass(frozen=True, slots=True)
class OnvifProbeOptions:
    host: str
    username: str
    password: str
    port: int = DEFAULT_ONVIF_PORT
    timeout_s: float | None = DEFAULT_PROBE_TIMEOUT_S
    wsdl_dir: str | None = None

    def normalized(self) -> OnvifProbeOptions:
        return OnvifProbeOptions(
            host=self.host.strip(),
            username=self.username.strip(),
            password=self.password.strip(),
            port=self.port,
            timeout_s=self.timeout_s,
            wsdl_dir=self.wsdl_dir,
        )


@dataclass(frozen=True, slots=True)
class OnvifProbeResult:
    device_info: OnvifDeviceInfo
    profiles: list[OnvifMediaProfile]
    streams: list[OnvifStreamUri]


class OnvifServiceError(RuntimeError):
    """Base ONVIF service error."""


class OnvifDiscoverError(OnvifServiceError):
    """Raised when discovery fails."""

    def __init__(self, detail: str, *, cause: Exception) -> None:
        super().__init__(detail)
        self.__cause__ = cause


class OnvifProbeError(OnvifServiceError):
    """Raised when probe operations fail for non-timeout reasons."""

    def __init__(self, detail: str, *, cause: Exception) -> None:
        super().__init__(detail)
        self.__cause__ = cause


class OnvifProbeTimeoutError(OnvifServiceError):
    """Raised when probe operations exceed the configured timeout."""

    def __init__(self, timeout_s: float, *, cause: Exception) -> None:
        super().__init__(f"ONVIF probe timed out after {timeout_s:.2f}s")
        self.timeout_s = timeout_s
        self.__cause__ = cause


_TReturn = TypeVar("_TReturn")


class _DiscoverFn(Protocol):
    def __call__(
        self,
        timeout_s: float,
        *,
        attempts: int,
        ttl: int,
    ) -> list[DiscoveredCamera]: ...


class _ClientFactory(Protocol):
    def __call__(
        self,
        host: str,
        username: str,
        password: str,
        *,
        port: int,
        wsdl_dir: str | None = None,
    ) -> OnvifCameraClient: ...


class OnvifService:
    """Coordinates ONVIF discovery and probe flows with consistent error handling."""

    def __init__(
        self,
        *,
        discover_fn: _DiscoverFn = discover_cameras,
        client_factory: _ClientFactory = OnvifCameraClient,
        close_timeout_s: float = DEFAULT_CLIENT_CLOSE_TIMEOUT_S,
        service_logger: logging.Logger | None = None,
    ) -> None:
        self._discover_fn = discover_fn
        self._client_factory = client_factory
        self._client_factory_supports_wsdl_dir = _supports_wsdl_dir(client_factory)
        self._close_timeout_s = close_timeout_s
        self._logger = service_logger or logger

    async def discover(self, options: OnvifDiscoverOptions) -> list[DiscoveredCamera]:
        try:
            return await asyncio.to_thread(
                self._discover_fn,
                options.timeout_s,
                attempts=options.attempts,
                ttl=options.ttl,
            )
        except Exception as exc:
            raise OnvifDiscoverError(str(exc), cause=exc) from exc

    async def fetch_device_and_profiles(
        self,
        options: OnvifProbeOptions,
    ) -> tuple[OnvifDeviceInfo, list[OnvifMediaProfile]]:
        async def _operation(
            client: OnvifCameraClient,
        ) -> tuple[OnvifDeviceInfo, list[OnvifMediaProfile]]:
            return await client.get_device_info(), await client.get_media_profiles()

        return await self._run_probe_operation(options.normalized(), _operation)

    async def fetch_stream_uris(self, options: OnvifProbeOptions) -> list[OnvifStreamUri]:
        async def _operation(client: OnvifCameraClient) -> list[OnvifStreamUri]:
            return await client.get_stream_uris()

        return await self._run_probe_operation(options.normalized(), _operation)

    async def probe(self, options: OnvifProbeOptions) -> OnvifProbeResult:
        async def _operation(client: OnvifCameraClient) -> OnvifProbeResult:
            device_info = await client.get_device_info()
            profiles = await client.get_media_profiles()
            streams = await client.get_stream_uris(profiles)
            return OnvifProbeResult(device_info=device_info, profiles=profiles, streams=streams)

        return await self._run_probe_operation(options.normalized(), _operation)

    async def _run_probe_operation(
        self,
        options: OnvifProbeOptions,
        operation: Callable[[OnvifCameraClient], Awaitable[_TReturn]],
    ) -> _TReturn:
        client: OnvifCameraClient | None = None
        try:
            if self._client_factory_supports_wsdl_dir:
                client = self._client_factory(
                    options.host,
                    options.username,
                    options.password,
                    port=options.port,
                    wsdl_dir=options.wsdl_dir,
                )
            else:
                client = self._client_factory(
                    options.host,
                    options.username,
                    options.password,
                    port=options.port,
                )
            if options.timeout_s is None:
                result = await operation(client)
            else:
                result = await asyncio.wait_for(
                    operation(client),
                    timeout=options.timeout_s,
                )
            return result
        except TimeoutError as exc:
            if options.timeout_s is None:
                raise OnvifProbeError(str(exc), cause=exc) from exc
            raise OnvifProbeTimeoutError(options.timeout_s, cause=exc) from exc
        except Exception as exc:
            raise OnvifProbeError(str(exc), cause=exc) from exc
        finally:
            if client is not None:
                await close_onvif_client_best_effort(
                    client,
                    timeout_s=self._close_timeout_s,
                    service_logger=self._logger,
                    context=f"host={options.host} port={options.port}",
                )


async def close_onvif_client_best_effort(
    client: OnvifCameraClient,
    *,
    timeout_s: float = DEFAULT_CLIENT_CLOSE_TIMEOUT_S,
    service_logger: logging.Logger | None = None,
    context: str | None = None,
) -> None:
    log = service_logger or logger
    suffix = f" ({context})" if context else ""
    try:
        await asyncio.wait_for(client.close(), timeout=timeout_s)
    except TimeoutError:
        log.warning(
            "ONVIF client close timed out after %.2fs%s",
            timeout_s,
            suffix,
            exc_info=True,
        )
    except Exception:
        log.warning("ONVIF client close failed%s", suffix, exc_info=True)


def _supports_wsdl_dir(client_factory: _ClientFactory) -> bool:
    try:
        signature = inspect.signature(client_factory)
    except (TypeError, ValueError):
        # Preserve compatibility when runtime introspection is unavailable.
        return True

    if "wsdl_dir" in signature.parameters:
        return True

    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


__all__ = [
    "DEFAULT_CLIENT_CLOSE_TIMEOUT_S",
    "DEFAULT_DISCOVER_ATTEMPTS",
    "DEFAULT_DISCOVER_TIMEOUT_S",
    "DEFAULT_DISCOVER_TTL",
    "DEFAULT_ONVIF_PORT",
    "DEFAULT_PROBE_TIMEOUT_S",
    "OnvifDiscoverError",
    "OnvifDiscoverOptions",
    "OnvifProbeError",
    "OnvifProbeOptions",
    "OnvifProbeResult",
    "OnvifProbeTimeoutError",
    "OnvifService",
    "close_onvif_client_best_effort",
]
