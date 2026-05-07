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
    # Given: A stream containing one length-prefixed frame followed by EOF
    reader = asyncio.StreamReader()
    reader.feed_data((5).to_bytes(4, "big") + b"hello")
    reader.feed_eof()

    # When: Reading frames from the stream
    first_frame = await read_length_prefixed_frame(reader, max_payload_bytes=10)
    second_frame = await read_length_prefixed_frame(reader, max_payload_bytes=10)

    # Then: The payload is returned once and EOF is represented as None
    assert first_frame == b"hello"
    assert second_frame is None


@pytest.mark.asyncio
async def test_read_length_prefixed_frame_rejects_oversized_payload() -> None:
    # Given: A stream frame declaring a payload larger than the configured limit
    reader = asyncio.StreamReader()
    reader.feed_data((11).to_bytes(4, "big"))
    reader.feed_eof()

    # When: Reading the frame from the IPC stream
    # Then: The helper fails before consuming an oversized payload
    with pytest.raises(IPCFrameError, match="exceeds"):
        await read_length_prefixed_frame(reader, max_payload_bytes=10)


@pytest.mark.asyncio
async def test_read_length_prefixed_frame_reports_truncated_payload() -> None:
    # Given: A stream frame whose declared payload is not fully delivered
    reader = asyncio.StreamReader()
    reader.feed_data((5).to_bytes(4, "big") + b"he")
    reader.feed_eof()

    # When: Reading the frame from the IPC stream
    # Then: The helper reports the truncated payload as EOF
    with pytest.raises(EOFError, match="payload"):
        await read_length_prefixed_frame(reader, max_payload_bytes=10)


@pytest.mark.asyncio
async def test_write_length_prefixed_frame() -> None:
    # Given: A stream writer that records all bytes written by the IPC helper
    written = bytearray()

    class _Writer:
        def write(self, data: bytes) -> None:
            written.extend(data)

        async def drain(self) -> None:
            return None

    # When: Writing a payload through the length-prefixed helper
    await write_length_prefixed_frame(_Writer(), b"hello")

    # Then: The writer receives a big-endian length prefix followed by the payload
    assert bytes(written) == (5).to_bytes(4, "big") + b"hello"
