"""Small multipart helpers for the Tapo local stream endpoint."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field

CLIENT_BOUNDARY = "--client-stream-boundary--"
CLIENT_PART_PREFIX = "--" + CLIENT_BOUNDARY
DEVICE_BOUNDARY = "--device-stream-boundary--"
DEVICE_PART_PREFIX = "--" + DEVICE_BOUNDARY

_CRLF = b"\r\n"


class TapoMultipartError(RuntimeError):
    """Raised for malformed Tapo multipart data."""


@dataclass(frozen=True, slots=True)
class TapoMultipartPart:
    """One parsed multipart part."""

    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    def header(self, name: str) -> str | None:
        """Return a case-insensitive header."""
        return self.headers.get(name.lower())


def multipart_part(
    boundary_prefix: str,
    headers: Mapping[str, str],
    body: bytes,
) -> bytes:
    """Render one multipart part with CRLF line endings."""
    rendered_headers = dict(headers)
    rendered_headers.setdefault("Content-Length", str(len(body)))
    lines = [boundary_prefix]
    lines.extend(f"{name}: {value}" for name, value in rendered_headers.items())
    head = "\r\n".join(lines).encode("iso-8859-1") + _CRLF * 2
    return head + body + _CRLF


async def read_multipart_part(
    reader: asyncio.StreamReader,
    *,
    boundary: str,
    max_payload_bytes: int,
    timeout_s: float,
) -> TapoMultipartPart:
    """Read one Content-Length bounded multipart part."""
    if max_payload_bytes < 0:
        raise ValueError("max_payload_bytes must be non-negative")
    delimiter = ("--" + boundary).encode("ascii")
    closing_delimiter = delimiter + b"--"

    while True:
        line = await _readline(reader, timeout_s=timeout_s)
        stripped = line.rstrip(b"\r\n")
        if stripped == delimiter:
            break
        if stripped == closing_delimiter:
            raise TapoMultipartError("Tapo multipart stream ended before next part")
        if stripped:
            continue

    headers: dict[str, str] = {}
    while True:
        line = await _readline(reader, timeout_s=timeout_s)
        if line in {b"\r\n", b"\n", b""}:
            break
        name, separator, value = line.decode("iso-8859-1").partition(":")
        if not separator:
            raise TapoMultipartError("Invalid Tapo multipart header")
        headers[name.strip().lower()] = value.strip()

    raw_content_length = headers.get("content-length")
    if raw_content_length is None:
        raise TapoMultipartError("Tapo multipart part missing Content-Length")
    try:
        content_length = int(raw_content_length)
    except ValueError as exc:
        raise TapoMultipartError("Invalid Tapo multipart Content-Length") from exc
    if content_length < 0:
        raise TapoMultipartError("Invalid Tapo multipart Content-Length")
    if content_length > max_payload_bytes:
        raise TapoMultipartError("Tapo multipart payload exceeds configured limit")

    body = await asyncio.wait_for(reader.readexactly(content_length), timeout=timeout_s)
    return TapoMultipartPart(headers=headers, body=body)


async def _readline(reader: asyncio.StreamReader, *, timeout_s: float) -> bytes:
    try:
        line = await asyncio.wait_for(reader.readline(), timeout=timeout_s)
    except asyncio.IncompleteReadError as exc:
        raise TapoMultipartError("Tapo multipart stream ended unexpectedly") from exc
    if line == b"":
        raise TapoMultipartError("Tapo multipart stream ended unexpectedly")
    return line
