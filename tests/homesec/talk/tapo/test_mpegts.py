"""Tests for the Tapo MPEG-TS PCMA muxer."""

from __future__ import annotations

import pytest

from homesec.talk.tapo.mpegts import TapoMPEGTSError, TapoPCMATransportStreamMuxer

_TS_PACKET_SIZE = 188
_TS_SYNC_BYTE = 0x47
_PAT_PID = 0x0000
_PMT_PID = 0x1000
_AUDIO_PID = 0x0100
_PROGRAM_NUMBER = 1
_STREAM_TYPE_PCMA_TAPO = 0x90
_AUDIO_STREAM_ID = 0xC0
_AUDIO_CLOCK_INCREMENT_20MS = 1800


def test_mpegts_header_contains_valid_pat_and_pmt_packets() -> None:
    """Muxer header should contain independently parseable PAT and PMT packets."""
    # Given: A fresh Tapo PCMA transport-stream muxer
    muxer = TapoPCMATransportStreamMuxer()

    # When: Rendering the stream header
    header = muxer.header()

    # Then: It contains a PAT mapping the program to the PMT and a PMT for audio
    packets = _packets(header)
    assert len(packets) == 2
    pat_packet, pmt_packet = packets
    assert _pid(pat_packet) == _PAT_PID
    assert _pid(pmt_packet) == _PMT_PID
    assert _payload_unit_start(pat_packet) is True
    assert _payload_unit_start(pmt_packet) is True
    assert _continuity_counter(pat_packet) == 0
    assert _continuity_counter(pmt_packet) == 0

    pat = _parse_pat(pat_packet)
    pmt = _parse_pmt(pmt_packet)
    assert pat == {"program_number": _PROGRAM_NUMBER, "pmt_pid": _PMT_PID}
    assert pmt == {
        "program_number": _PROGRAM_NUMBER,
        "pcr_pid": _AUDIO_PID,
        "stream_type": _STREAM_TYPE_PCMA_TAPO,
        "elementary_pid": _AUDIO_PID,
    }


def test_mpegts_audio_payload_packets_have_pes_pts_and_pcr() -> None:
    """Audio payload output should be packetized as timed MPEG-TS packets."""
    # Given: One 20 ms PCMA frame
    muxer = TapoPCMATransportStreamMuxer()
    pcma = bytes(range(160))

    # When: Packetizing audio
    payload = muxer.audio_payload(pcma)

    # Then: The first audio packet has PES, PTS, and PCR for the advertised PCR PID
    packets = _packets(payload)
    assert len(packets) == 1
    packet = packets[0]
    assert _pid(packet) == _AUDIO_PID
    assert _payload_unit_start(packet) is True
    assert _continuity_counter(packet) == 0
    assert _pcr_base(packet) == 0

    pes = _payload(packet)
    assert pes[:4] == b"\x00\x00\x01" + bytes([_AUDIO_STREAM_ID])
    assert int.from_bytes(pes[4:6], "big") == len(pes) - 6
    assert pes[6:9] == b"\x80\x80\x05"
    assert _pts_from_pes(pes) == 0
    assert pes[14:] == pcma


def test_mpegts_audio_continuity_counter_advances_across_multiple_packets() -> None:
    """Audio continuity counters should advance for multi-packet payloads."""
    # Given: A PCMA payload large enough to span multiple TS packets
    muxer = TapoPCMATransportStreamMuxer()
    pcma = bytes([0xAA]) * 400

    # When: Packetizing audio
    packets = _packets(muxer.audio_payload(pcma))

    # Then: Counters advance and only the first packet starts the PES/PCR sequence
    assert len(packets) == 3
    assert [_continuity_counter(packet) for packet in packets] == [0, 1, 2]
    assert [_payload_unit_start(packet) for packet in packets] == [True, False, False]
    assert _pcr_base(packets[0]) == 0
    assert _pcr_base(packets[1]) is None
    assert _pcr_base(packets[2]) is None


def test_mpegts_audio_pts_uses_90khz_clock() -> None:
    """Audio PTS should advance on the MPEG-TS 90 kHz clock."""
    # Given: A muxer with deterministic initial timestamp
    muxer = TapoPCMATransportStreamMuxer()

    # When: Emitting two 160-sample PCMA frames
    first = _packets(muxer.audio_payload(b"\xaa" * 160))[0]
    second = _packets(muxer.audio_payload(b"\xbb" * 160))[0]

    # Then: The second PES timestamp advances by 20 ms at 90 kHz
    assert _pts_from_pes(_payload(first)) == 0
    assert _pts_from_pes(_payload(second)) == _AUDIO_CLOCK_INCREMENT_20MS
    assert _pcr_base(second) == _AUDIO_CLOCK_INCREMENT_20MS


def test_mpegts_output_is_deterministic_for_fixed_input() -> None:
    """Fresh muxers should produce identical output for identical PCMA input."""
    # Given: Two fresh muxers and fixed PCMA data
    pcma = b"\xd5" * 160

    # When: Rendering headers and one audio payload from each muxer
    first_muxer = TapoPCMATransportStreamMuxer()
    first = first_muxer.header() + first_muxer.audio_payload(pcma)
    muxer = TapoPCMATransportStreamMuxer()
    second = muxer.header() + muxer.audio_payload(pcma)

    # Then: The output is deterministic
    assert first == second


def test_mpegts_rejects_empty_audio_payload() -> None:
    """Muxer should reject empty audio payloads."""
    # Given: A fresh muxer
    muxer = TapoPCMATransportStreamMuxer()

    # When/Then: Packetizing an empty payload fails
    with pytest.raises(TapoMPEGTSError, match="must not be empty"):
        muxer.audio_payload(b"")


def _packets(data: bytes) -> list[bytes]:
    assert len(data) % _TS_PACKET_SIZE == 0
    packets = [
        data[index : index + _TS_PACKET_SIZE] for index in range(0, len(data), _TS_PACKET_SIZE)
    ]
    assert all(len(packet) == _TS_PACKET_SIZE for packet in packets)
    assert all(packet[0] == _TS_SYNC_BYTE for packet in packets)
    return packets


def _pid(packet: bytes) -> int:
    return ((packet[1] & 0x1F) << 8) | packet[2]


def _payload_unit_start(packet: bytes) -> bool:
    return bool(packet[1] & 0x40)


def _continuity_counter(packet: bytes) -> int:
    return packet[3] & 0x0F


def _payload(packet: bytes) -> bytes:
    adaptation_control = (packet[3] >> 4) & 0x03
    offset = 4
    if adaptation_control & 0x02:
        offset += 1 + packet[offset]
    assert adaptation_control & 0x01
    payload = packet[offset:]
    if _payload_unit_start(packet) and _pid(packet) in {_PAT_PID, _PMT_PID}:
        pointer = payload[0]
        payload = payload[1 + pointer :]
    return payload


def _section(packet: bytes, *, table_id: int) -> bytes:
    payload = _payload(packet)
    assert payload[0] == table_id
    section_length = ((payload[1] & 0x0F) << 8) | payload[2]
    section = payload[: 3 + section_length]
    assert len(section) == 3 + section_length
    assert _mpeg2_crc32(section[:-4]) == int.from_bytes(section[-4:], "big")
    return section


def _parse_pat(packet: bytes) -> dict[str, int]:
    section = _section(packet, table_id=0x00)
    program_number = int.from_bytes(section[8:10], "big")
    pmt_pid = ((section[10] & 0x1F) << 8) | section[11]
    return {"program_number": program_number, "pmt_pid": pmt_pid}


def _parse_pmt(packet: bytes) -> dict[str, int]:
    section = _section(packet, table_id=0x02)
    program_number = int.from_bytes(section[3:5], "big")
    pcr_pid = ((section[8] & 0x1F) << 8) | section[9]
    program_info_length = ((section[10] & 0x0F) << 8) | section[11]
    offset = 12 + program_info_length
    stream_type = section[offset]
    elementary_pid = ((section[offset + 1] & 0x1F) << 8) | section[offset + 2]
    es_info_length = ((section[offset + 3] & 0x0F) << 8) | section[offset + 4]
    assert es_info_length == 0
    assert offset + 5 == len(section) - 4
    return {
        "program_number": program_number,
        "pcr_pid": pcr_pid,
        "stream_type": stream_type,
        "elementary_pid": elementary_pid,
    }


def _pts_from_pes(pes: bytes) -> int:
    pts = pes[9:14]
    assert len(pts) == 5
    assert (pts[0] & 0xF0) == 0x20
    assert pts[0] & 0x01
    assert pts[2] & 0x01
    assert pts[4] & 0x01
    return (
        ((pts[0] >> 1) & 0x07) << 30
        | pts[1] << 22
        | ((pts[2] >> 1) & 0x7F) << 15
        | pts[3] << 7
        | ((pts[4] >> 1) & 0x7F)
    )


def _pcr_base(packet: bytes) -> int | None:
    adaptation_control = (packet[3] >> 4) & 0x03
    if not adaptation_control & 0x02:
        return None
    adaptation_length = packet[4]
    if adaptation_length == 0:
        return None
    adaptation = packet[5 : 5 + adaptation_length]
    flags = adaptation[0]
    if not flags & 0x10:
        return None
    raw = int.from_bytes(adaptation[1:7], "big")
    assert ((raw >> 9) & 0x3F) == 0x3F
    assert (raw & 0x1FF) == 0
    return raw >> 15


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
