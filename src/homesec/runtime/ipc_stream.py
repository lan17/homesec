"""Length-prefixed IPC stream helpers used by runtime media bridges."""

from __future__ import annotations

import asyncio
from typing import Protocol

_FRAME_HEADER_BYTES = 4


class _AsyncReader(Protocol):
    async def readexactly(self, n: int) -> bytes: ...


class _AsyncWriter(Protocol):
    def write(self, data: bytes) -> None: ...

    async def drain(self) -> None: ...


class IPCFrameError(ValueError):
    """Raised when an IPC media frame is malformed or over limit."""


async def read_length_prefixed_frame(
    reader: _AsyncReader,
    *,
    max_payload_bytes: int,
) -> bytes | None:
    """Read one uint32_be length-prefixed frame.

    Returns None on clean EOF before the frame header. Raises EOFError on a
    truncated header/payload and IPCFrameError when the payload exceeds the
    configured bound.
    """
    if max_payload_bytes <= 0:
        raise ValueError("max_payload_bytes must be positive")
    try:
        header = await reader.readexactly(_FRAME_HEADER_BYTES)
    except asyncio.IncompleteReadError as exc:
        if exc.partial == b"":
            return None
        raise EOFError("truncated IPC frame header") from exc

    payload_len = int.from_bytes(header, "big")
    if payload_len > max_payload_bytes:
        raise IPCFrameError(
            f"IPC frame exceeds max payload size: {payload_len} > {max_payload_bytes}"
        )
    try:
        return await reader.readexactly(payload_len)
    except asyncio.IncompleteReadError as exc:
        raise EOFError("truncated IPC frame payload") from exc


async def write_length_prefixed_frame(writer: _AsyncWriter, payload: bytes) -> None:
    """Write one uint32_be length-prefixed frame and drain the stream."""
    if len(payload) > 0xFFFFFFFF:
        raise IPCFrameError("IPC frame exceeds uint32 length prefix")
    writer.write(len(payload).to_bytes(_FRAME_HEADER_BYTES, "big") + payload)
    await writer.drain()
