"""Tests for ONVIF standalone CLI command handlers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from homesec.onvif.cli import OnvifCLI, _parse_host_port
from homesec.onvif.client import OnvifDeviceInfo, OnvifMediaProfile, OnvifStreamUri
from homesec.onvif.discovery import DiscoveredCamera


def test_onvif_cli_discover_prints_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """OnvifCLI.discover should print discovered devices."""
    # Given: Discovery returns one ONVIF endpoint
    monkeypatch.setattr(
        "homesec.onvif.cli.discover_cameras",
        lambda timeout_s, attempts, ttl: [
            DiscoveredCamera(
                ip="192.168.1.20",
                xaddr="http://192.168.1.20/onvif/device_service",
                scopes=("onvif://scope/location/garage",),
                types=("dn:NetworkVideoTransmitter",),
            )
        ],
    )

    # When: Running discover command
    cli = OnvifCLI()
    cli.discover(timeout_s=2.0, attempts=1)

    # Then: CLI output includes endpoint identity and metadata
    captured = capsys.readouterr()
    assert "Discovered ONVIF cameras" in captured.out
    assert "192.168.1.20" in captured.out
    assert "NetworkVideoTransmitter" in captured.out


def test_onvif_cli_discover_prints_debug_tips_when_no_cameras_found(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """OnvifCLI.discover should provide operator guidance when nothing is discovered."""
    # Given: Discovery returns no endpoints
    monkeypatch.setattr("homesec.onvif.cli.discover_cameras", lambda timeout_s, attempts, ttl: [])

    # When: Running discover command
    cli = OnvifCLI()
    cli.discover(timeout_s=2.0, attempts=1)

    # Then: CLI output includes concrete next-step hints
    captured = capsys.readouterr()
    assert "No ONVIF cameras discovered." in captured.out
    assert "Verify ONVIF and WS-Discovery are enabled" in captured.out
    assert "--timeout_s 15 --attempts 3" in captured.out


def test_onvif_cli_info_prints_device_and_profiles(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """OnvifCLI.info should print device metadata and media profile summary."""

    @dataclass
    class _FakeClient:
        ip: str
        username: str
        password: str
        port: int
        wsdl_dir: str | None

        async def get_device_info(self) -> OnvifDeviceInfo:
            return OnvifDeviceInfo(
                manufacturer="Acme",
                model="ModelX",
                firmware_version="1.0.0",
                serial_number="ABC123",
                hardware_id="HW1",
            )

        async def get_media_profiles(self) -> list[OnvifMediaProfile]:
            return [
                OnvifMediaProfile(
                    token="main",
                    name="Main",
                    video_encoding="H265",
                    width=2560,
                    height=1440,
                    frame_rate_limit=20,
                    bitrate_limit_kbps=6144,
                )
            ]

        async def close(self) -> None:
            pass

    # Given: ONVIF client wrapper returns deterministic metadata
    monkeypatch.setattr("homesec.onvif.cli.OnvifCameraClient", _FakeClient)

    # When: Running info command
    cli = OnvifCLI()
    cli.info("192.168.1.30", "admin", "secret", port=8080, wsdl_dir="/wsdl")

    # Then: CLI output includes device and profile details
    captured = capsys.readouterr()
    assert "ONVIF device at 192.168.1.30:8080" in captured.out
    assert "manufacturer: Acme" in captured.out
    assert "token=main name=Main" in captured.out


def test_onvif_cli_streams_exits_with_error_when_client_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """OnvifCLI.streams should exit non-zero and print error on client failure."""

    class _FailingClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            _ = args
            _ = kwargs

        async def get_stream_uris(self) -> list[OnvifStreamUri]:
            raise RuntimeError("invalid credentials")

        async def close(self) -> None:
            pass

    # Given: ONVIF client fails to fetch streams
    monkeypatch.setattr("homesec.onvif.cli.OnvifCameraClient", _FailingClient)

    # When/Then: Running streams command exits with code 1 and prints error
    cli = OnvifCLI()
    with pytest.raises(SystemExit) as exc_info:
        cli.streams("192.168.1.40", "admin", "bad-password")

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "invalid credentials" in captured.err


def test_parse_host_port_supports_bare_ipv6_literal() -> None:
    """_parse_host_port should not misinterpret bare IPv6 literals as host:port."""
    # Given: A bare IPv6 literal and default port
    addr = "fe80::1"

    # When: Parsing address
    host, port = _parse_host_port(addr, 80)

    # Then: Host remains the full IPv6 literal and default port is preserved
    assert host == "fe80::1"
    assert port == 80


def test_parse_host_port_supports_bracketed_ipv6_with_port() -> None:
    """_parse_host_port should parse bracketed IPv6 literals with explicit port."""
    # Given: A bracketed IPv6 host with explicit port
    addr = "[fe80::1]:8899"

    # When: Parsing address
    host, port = _parse_host_port(addr, 80)

    # Then: Brackets are removed and port override is applied
    assert host == "fe80::1"
    assert port == 8899
