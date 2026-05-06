"""RTP packetization helpers for camera talk backchannels."""

from __future__ import annotations

import secrets
import struct
from dataclasses import dataclass

_RTP_VERSION = 2
_RTP_FIXED_HEADER_LEN = 12


@dataclass(slots=True)
class RTPPacketizer:
    """Build sequential RTP packets for one audio payload stream."""

    payload_type: int
    clock_rate: int = 8000
    sequence_number: int | None = None
    timestamp: int | None = None
    ssrc: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.payload_type <= 127:
            raise ValueError("RTP payload type must be in range 0..127")
        if self.clock_rate <= 0:
            raise ValueError("RTP clock_rate must be positive")
        if self.sequence_number is None:
            self.sequence_number = secrets.randbits(16)
        if self.timestamp is None:
            self.timestamp = secrets.randbits(32)
        if self.ssrc is None:
            self.ssrc = secrets.randbits(32)
        self.sequence_number &= 0xFFFF
        self.timestamp &= 0xFFFFFFFF
        self.ssrc &= 0xFFFFFFFF

    def packetize(
        self,
        payload: bytes,
        *,
        timestamp_increment: int | None = None,
        marker: bool = False,
    ) -> bytes:
        """Return an RTP packet and advance sequence/timestamp counters."""
        if timestamp_increment is None:
            timestamp_increment = len(payload)
        if timestamp_increment < 0:
            raise ValueError("timestamp_increment must be non-negative")

        assert self.sequence_number is not None
        assert self.timestamp is not None
        assert self.ssrc is not None

        first = (_RTP_VERSION << 6) & 0xC0
        second = self.payload_type | (0x80 if marker else 0)
        header = struct.pack(
            "!BBHII",
            first,
            second,
            self.sequence_number,
            self.timestamp,
            self.ssrc,
        )
        packet = header + payload
        self.sequence_number = (self.sequence_number + 1) & 0xFFFF
        self.timestamp = (self.timestamp + timestamp_increment) & 0xFFFFFFFF
        return packet


def parse_rtp_header(packet: bytes) -> dict[str, int | bool]:
    """Parse the fixed RTP header fields used by tests and fake camera capture."""
    if len(packet) < _RTP_FIXED_HEADER_LEN:
        raise ValueError("RTP packet shorter than fixed header")
    first, second, sequence, timestamp, ssrc = struct.unpack("!BBHII", packet[:12])
    return {
        "version": first >> 6,
        "padding": bool(first & 0x20),
        "extension": bool(first & 0x10),
        "csrc_count": first & 0x0F,
        "marker": bool(second & 0x80),
        "payload_type": second & 0x7F,
        "sequence_number": sequence,
        "timestamp": timestamp,
        "ssrc": ssrc,
    }
