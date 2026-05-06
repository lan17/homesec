from __future__ import annotations

import asyncio

import pytest

from homesec.runtime.ipc_stream import (
    IPCFrameError,
    read_length_prefixed_frame,
    write_length_prefixed_frame,
)


@pytest.mark.asyncio
async def test_read_length_prefixed_frame_reads_payload_and_eof() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data((5).to_bytes(4, "big") + b"hello")
    reader.feed_eof()

    assert await read_length_prefixed_frame(reader, max_payload_bytes=10) == b"hello"
    assert await read_length_prefixed_frame(reader, max_payload_bytes=10) is None


@pytest.mark.asyncio
async def test_read_length_prefixed_frame_rejects_oversized_payload() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data((11).to_bytes(4, "big"))
    reader.feed_eof()

    with pytest.raises(IPCFrameError, match="exceeds"):
        await read_length_prefixed_frame(reader, max_payload_bytes=10)


@pytest.mark.asyncio
async def test_read_length_prefixed_frame_reports_truncated_payload() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data((5).to_bytes(4, "big") + b"he")
    reader.feed_eof()

    with pytest.raises(EOFError, match="payload"):
        await read_length_prefixed_frame(reader, max_payload_bytes=10)


@pytest.mark.asyncio
async def test_write_length_prefixed_frame() -> None:
    written = bytearray()

    class _Writer:
        def write(self, data: bytes) -> None:
            written.extend(data)

        async def drain(self) -> None:
            return None

    await write_length_prefixed_frame(_Writer(), b"hello")

    assert bytes(written) == (5).to_bytes(4, "big") + b"hello"
