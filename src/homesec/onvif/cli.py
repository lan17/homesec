"""CLI for ONVIF discovery and probing."""

from __future__ import annotations

import asyncio
import getpass
import sys

import fire  # type: ignore[import-untyped]

from homesec.onvif.client import (
    OnvifCameraClient,
    OnvifDeviceInfo,
    OnvifMediaProfile,
    OnvifStreamUri,
)
from homesec.onvif.discovery import discover_cameras


class OnvifCLI:
    """Standalone ONVIF utilities."""

    def discover(self, timeout_s: float = 8.0, attempts: int = 2, ttl: int = 4) -> None:
        """Discover ONVIF devices on the local network."""
        try:
            cameras = discover_cameras(timeout_s=timeout_s, attempts=attempts, ttl=ttl)
        except Exception as exc:
            _exit_with_error(str(exc))
            return

        if not cameras:
            print("No ONVIF cameras discovered.")
            print("Tips:")
            print("- Verify ONVIF and WS-Discovery are enabled on the camera.")
            print(
                "- Ensure HomeSec host and camera are on the same L2 subnet (multicast required)."
            )
            print("- Retry with a longer scan: --timeout_s 15 --attempts 3")
            print("- If camera IP is known, probe directly with: info <ip> -u <user>")
            return

        print("Discovered ONVIF cameras:")
        for camera in cameras:
            print(f"- ip={camera.ip} xaddr={camera.xaddr}")
            if camera.types:
                print(f"  types: {', '.join(camera.types)}")
            if camera.scopes:
                print(f"  scopes: {', '.join(camera.scopes)}")

    def info(
        self,
        ip: str,
        u: str,
        p: str | None = None,
        port: int = 80,
        wsdl_dir: str | None = None,
    ) -> None:
        """Show device information and media profile metadata.

        The ip argument accepts 'host' or 'host:port' (overrides --port).
        """
        host, resolved_port = _parse_host_port(ip, port)
        password = p if p is not None else getpass.getpass("ONVIF password: ")

        async def _run() -> tuple[OnvifDeviceInfo, list[OnvifMediaProfile]]:
            client = OnvifCameraClient(
                host, u, password, port=resolved_port, wsdl_dir=wsdl_dir
            )
            try:
                return await client.get_device_info(), await client.get_media_profiles()
            finally:
                await client.close()

        try:
            info, profiles = asyncio.run(_run())
        except Exception as exc:
            _exit_with_error(str(exc))
            return

        print(f"ONVIF device at {host}:{resolved_port}")
        print(f"  manufacturer: {info.manufacturer}")
        print(f"  model: {info.model}")
        print(f"  firmware_version: {info.firmware_version}")
        print(f"  serial_number: {info.serial_number}")
        print(f"  hardware_id: {info.hardware_id}")

        if not profiles:
            print("No media profiles reported.")
            return

        print("Media profiles:")
        for profile in profiles:
            print(f"- token={profile.token} name={profile.name}")
            print(
                "  video:"
                f" encoding={profile.video_encoding or 'unknown'}"
                f" width={profile.width or 'unknown'}"
                f" height={profile.height or 'unknown'}"
                f" fps_limit={profile.frame_rate_limit or 'unknown'}"
                f" bitrate_kbps={profile.bitrate_limit_kbps or 'unknown'}"
            )

    def streams(
        self,
        ip: str,
        u: str,
        p: str | None = None,
        port: int = 80,
        wsdl_dir: str | None = None,
    ) -> None:
        """Show RTSP stream URIs for each ONVIF media profile.

        The ip argument accepts 'host' or 'host:port' (overrides --port).
        """
        host, resolved_port = _parse_host_port(ip, port)
        password = p if p is not None else getpass.getpass("ONVIF password: ")

        async def _run() -> list[OnvifStreamUri]:
            client = OnvifCameraClient(
                host, u, password, port=resolved_port, wsdl_dir=wsdl_dir
            )
            try:
                return await client.get_stream_uris()
            finally:
                await client.close()

        try:
            streams = asyncio.run(_run())
        except Exception as exc:
            _exit_with_error(str(exc))
            return

        if not streams:
            print("No media profiles found.")
            return

        print(f"RTSP streams for ONVIF device {host}:{resolved_port}:")
        for stream in streams:
            if stream.uri:
                print(f"- token={stream.profile_token} name={stream.profile_name} uri={stream.uri}")
                continue
            print(
                "-"
                f" token={stream.profile_token} name={stream.profile_name}"
                f" error={stream.error or 'unknown'}"
            )


def _parse_host_port(addr: str, default_port: int) -> tuple[str, int]:
    """Parse 'host' or 'host:port' into (host, port), including IPv6 literals."""
    normalized_addr = addr.strip()
    if normalized_addr.startswith("["):
        closing_idx = normalized_addr.find("]")
        if closing_idx != -1:
            host = normalized_addr[1:closing_idx]
            remainder = normalized_addr[closing_idx + 1 :]
            if remainder == "":
                return host, default_port
            if remainder.startswith(":"):
                port_str = remainder[1:]
                try:
                    return host, int(port_str)
                except ValueError:
                    return normalized_addr, default_port
            return normalized_addr, default_port

    # Treat values with more than one colon as bare IPv6 literals unless
    # bracketed (handled above).
    if normalized_addr.count(":") == 1:
        host, port_str = normalized_addr.rsplit(":", 1)
        try:
            return host, int(port_str)
        except ValueError:
            return normalized_addr, default_port
    return normalized_addr, default_port


def _exit_with_error(message: str) -> None:
    print(f"✗ {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    """ONVIF CLI entrypoint."""
    fire.Fire(OnvifCLI)


if __name__ == "__main__":
    main()
