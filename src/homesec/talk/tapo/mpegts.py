"""Minimal MPEG-TS muxer for Tapo PCMA talk audio."""

from __future__ import annotations

TS_PACKET_SIZE = 188
TS_SYNC_BYTE = 0x47

PAT_PID = 0x0000
PMT_PID = 0x1000
AUDIO_PID = 0x0100

STREAM_TYPE_PCMA_TAPO = 0x90
STREAM_TYPE_PCMU_TAPO = 0x91
AUDIO_STREAM_ID = 0xC0

_PROGRAM_NUMBER = 1
_TRANSPORT_STREAM_ID = 1
_MAX_TS_PAYLOAD_SIZE = 184
_PCR_ADAPTATION_BYTES = 8
_AUDIO_SAMPLE_RATE_HZ = 8_000
_MPEGTS_CLOCK_HZ = 90_000
_PTS_MASK = (1 << 33) - 1


class TapoMPEGTSError(ValueError):
    """Raised when MPEG-TS packetization input is invalid."""


class TapoPCMATransportStreamMuxer:
    """Build deterministic MPEG-TS packets for Tapo PCMA/8000 audio."""

    def __init__(self) -> None:
        self._continuity: dict[int, int] = {}
        self._pts = 0
        self._pts_remainder = 0

    def header(self) -> bytes:
        """Return PAT and PMT packets for the Tapo PCMA stream."""
        return self._section_packet(PAT_PID, _pat_section()) + self._section_packet(
            PMT_PID,
            _pmt_section(stream_type=STREAM_TYPE_PCMA_TAPO),
        )

    def audio_payload(self, pcma: bytes) -> bytes:
        """Return MPEG-TS packets containing one PCMA audio payload."""
        if not pcma:
            raise TapoMPEGTSError("PCMA payload must not be empty")
        pts = self._pts
        self._advance_pts(len(pcma))
        return self._pes_packets(AUDIO_PID, _pes_packet(pcma, pts=pts), pcr_base=pts)

    def _advance_pts(self, sample_count: int) -> None:
        units = (sample_count * _MPEGTS_CLOCK_HZ) + self._pts_remainder
        increment, self._pts_remainder = divmod(units, _AUDIO_SAMPLE_RATE_HZ)
        self._pts = (self._pts + increment) & _PTS_MASK

    def _section_packet(self, pid: int, section: bytes) -> bytes:
        return self._packetize_payload(pid, b"\x00" + section, payload_unit_start=True)

    def _pes_packets(self, pid: int, pes: bytes, *, pcr_base: int) -> bytes:
        return self._packetize_payload(
            pid,
            pes,
            payload_unit_start=True,
            first_packet_pcr_base=pcr_base,
        )

    def _packetize_payload(
        self,
        pid: int,
        payload: bytes,
        *,
        payload_unit_start: bool,
        first_packet_pcr_base: int | None = None,
    ) -> bytes:
        packets: list[bytes] = []
        first = True
        offset = 0
        while offset < len(payload):
            pcr_base = first_packet_pcr_base if first else None
            max_payload_size = _max_payload_size(pcr_base=pcr_base)
            remaining = len(payload) - offset
            if remaining >= max_payload_size:
                chunk = payload[offset : offset + max_payload_size]
                offset += len(chunk)
                packets.append(
                    self._ts_packet(
                        pid=pid,
                        payload=chunk,
                        payload_unit_start=payload_unit_start and first,
                        pcr_base=pcr_base,
                    )
                )
            else:
                chunk = payload[offset:]
                offset = len(payload)
                packets.append(
                    self._ts_packet(
                        pid=pid,
                        payload=chunk,
                        payload_unit_start=payload_unit_start and first,
                        pcr_base=pcr_base,
                        pad=True,
                    )
                )
            first = False
        return b"".join(packets)

    def _ts_packet(
        self,
        *,
        pid: int,
        payload: bytes,
        payload_unit_start: bool,
        pcr_base: int | None = None,
        pad: bool = False,
    ) -> bytes:
        max_payload_size = _max_payload_size(pcr_base=pcr_base)
        if len(payload) > max_payload_size:
            raise TapoMPEGTSError("TS payload exceeds one packet")
        continuity = self._continuity.get(pid, 0) & 0x0F
        self._continuity[pid] = (continuity + 1) & 0x0F

        second = ((0x40 if payload_unit_start else 0x00) | ((pid >> 8) & 0x1F)) & 0xFF
        if pcr_base is None and not pad and len(payload) == _MAX_TS_PAYLOAD_SIZE:
            header = bytes([TS_SYNC_BYTE, second, pid & 0xFF, 0x10 | continuity])
            packet = header + payload
        else:
            adaptation_length = _MAX_TS_PAYLOAD_SIZE - 1 - len(payload)
            if adaptation_length < 0:
                raise TapoMPEGTSError("TS payload cannot be padded")
            header = bytes([TS_SYNC_BYTE, second, pid & 0xFF, 0x30 | continuity])
            if adaptation_length == 0:
                if pcr_base is not None:
                    raise TapoMPEGTSError("TS packet cannot fit PCR adaptation data")
                adaptation = b"\x00"
            else:
                flags = 0x10 if pcr_base is not None else 0x00
                pcr = _encode_pcr(pcr_base) if pcr_base is not None else b""
                stuffing_length = adaptation_length - 1 - len(pcr)
                if stuffing_length < 0:
                    raise TapoMPEGTSError("TS packet cannot fit PCR adaptation data")
                adaptation = bytes([adaptation_length, flags]) + pcr + (b"\xff" * stuffing_length)
            packet = header + adaptation + payload
        if len(packet) != TS_PACKET_SIZE:
            raise TapoMPEGTSError("Internal TS packet sizing error")
        return packet


def _pat_section() -> bytes:
    body = (
        _TRANSPORT_STREAM_ID.to_bytes(2, "big")
        + b"\xc1\x00\x00"
        + _PROGRAM_NUMBER.to_bytes(2, "big")
        + bytes([0xE0 | ((PMT_PID >> 8) & 0x1F), PMT_PID & 0xFF])
    )
    section = _section_header(table_id=0x00, body=body)
    return section + _mpeg2_crc32(section).to_bytes(4, "big")


def _pmt_section(*, stream_type: int) -> bytes:
    body = (
        _PROGRAM_NUMBER.to_bytes(2, "big")
        + b"\xc1\x00\x00"
        + bytes([0xE0 | ((AUDIO_PID >> 8) & 0x1F), AUDIO_PID & 0xFF])
        + b"\xf0\x00"
        + bytes([stream_type, 0xE0 | ((AUDIO_PID >> 8) & 0x1F), AUDIO_PID & 0xFF])
        + b"\xf0\x00"
    )
    section = _section_header(table_id=0x02, body=body)
    return section + _mpeg2_crc32(section).to_bytes(4, "big")


def _section_header(*, table_id: int, body: bytes) -> bytes:
    section_length = len(body) + 4
    return bytes([table_id, 0xB0 | ((section_length >> 8) & 0x0F), section_length & 0xFF]) + body


def _pes_packet(payload: bytes, *, pts: int) -> bytes:
    header_data = _encode_pts(pts)
    pes_header = bytes([0x80, 0x80, len(header_data)]) + header_data
    packet_length = len(pes_header) + len(payload)
    if packet_length > 0xFFFF:
        packet_length = 0
    return (
        b"\x00\x00\x01"
        + bytes([AUDIO_STREAM_ID])
        + packet_length.to_bytes(2, "big")
        + pes_header
        + payload
    )


def _encode_pts(pts: int) -> bytes:
    pts &= _PTS_MASK
    return bytes(
        [
            (0x2 << 4) | (((pts >> 30) & 0x07) << 1) | 0x01,
            (pts >> 22) & 0xFF,
            (((pts >> 15) & 0x7F) << 1) | 0x01,
            (pts >> 7) & 0xFF,
            ((pts & 0x7F) << 1) | 0x01,
        ]
    )


def _encode_pcr(pcr_base: int) -> bytes:
    pcr_base &= _PTS_MASK
    return ((pcr_base << 15) | (0x3F << 9)).to_bytes(6, "big")


def _max_payload_size(*, pcr_base: int | None = None) -> int:
    if pcr_base is None:
        return _MAX_TS_PAYLOAD_SIZE
    return _MAX_TS_PAYLOAD_SIZE - _PCR_ADAPTATION_BYTES


def _mpeg2_crc32(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
    return crc
