"""Tests for shared ONVIF service orchestration."""

from __future__ import annotations

from typing import Any

import pytest

from homesec.onvif.service import OnvifProbeError, OnvifProbeOptions, OnvifService


async def test_probe_none_timeout_maps_operation_timeout_to_probe_error() -> None:
    """OnvifService should classify operation TimeoutError as OnvifProbeError when timeout is disabled."""

    class _TimeoutClient:
        closed = False

        async def get_device_info(self) -> Any:
            raise TimeoutError("camera operation timed out")

        async def get_media_profiles(self) -> list[Any]:
            return []

        async def close(self) -> None:
            self.closed = True

    client = _TimeoutClient()

    def _client_factory(
        host: str,
        username: str,
        password: str,
        *,
        port: int,
        wsdl_dir: str | None = None,
    ) -> _TimeoutClient:
        _ = host, username, password, port, wsdl_dir
        return client

    # Given: A service probe operation where timeout enforcement is disabled
    service = OnvifService(client_factory=_client_factory)

    # When/Then: A TimeoutError from the underlying operation maps to OnvifProbeError
    with pytest.raises(OnvifProbeError, match="camera operation timed out"):
        await service.fetch_device_and_profiles(
            OnvifProbeOptions(
                host="192.168.1.10",
                username="admin",
                password="secret",
                timeout_s=None,
            )
        )
    assert client.closed is True
