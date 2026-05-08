"""Tests for the Tapo MPEG-TS PCMA muxer."""

from __future__ import annotations

import pytest

from homesec.talk.tapo.mpegts import (
    AUDIO_PID,
    PAT_PID,
    PMT_PID,
    STREAM_TYPE_PCMA_TAPO,
    TS_PACKET_SIZE,
    TS_SYNC_BYTE,
    TapoMPEGTSError,
    TapoPCMATransportStreamMuxer,
)


def test_mpegts_header_contains_pat_and_pmt_packets() -> None:
    """Muxer header should contain deterministic PAT and PMT packets."""
    # Given: A fresh Tapo PCMA transport-stream muxer
    muxer = TapoPCMATransportStreamMuxer()

    # When: Rendering the stream header
    header = muxer.header()

    # Then: It contains two valid TS packets for PAT and PMT
    packets = _packets(header)
    assert len(packets) == 2
    assert all(packet[0] == TS_SYNC_BYTE for packet in packets)
    assert _pid(packets[0]) == PAT_PID
    assert _pid(packets[1]) == PMT_PID
    assert _continuity_counter(packets[0]) == 0
    assert _continuity_counter(packets[1]) == 0


def test_mpegts_pmt_contains_tapo_pcma_stream_type_and_audio_pid() -> None:
    """PMT should advertise Tapo's private PCMA stream type and audio PID."""
    # Given: A rendered header
    muxer = TapoPCMATransportStreamMuxer()
    _pat, pmt = _packets(muxer.header())

    # When: Extracting PMT payload bytes
    payload = _payload(pmt)

    # Then: The PMT advertises stream type 0x90 on PID 0x0100
    assert (
        bytes(
            [
                STREAM_TYPE_PCMA_TAPO,
                0xE0 | ((AUDIO_PID >> 8) & 0x1F),
                AUDIO_PID & 0xFF,
                0xF0,
                0x00,
            ]
        )
        in payload
    )


def test_mpegts_audio_payload_packets_are_188_byte_ts_packets() -> None:
    """Audio payload output should be packetized as MPEG-TS packets."""
    # Given: One 20 ms PCMA frame
    muxer = TapoPCMATransportStreamMuxer()
    pcma = bytes(range(160))

    # When: Packetizing audio
    payload = muxer.audio_payload(pcma)

    # Then: Every packet has the TS size, sync byte, and audio PID
    packets = _packets(payload)
    assert packets
    assert all(len(packet) == TS_PACKET_SIZE for packet in packets)
    assert all(packet[0] == TS_SYNC_BYTE for packet in packets)
    assert all(_pid(packet) == AUDIO_PID for packet in packets)
    assert _payload_unit_start(packets[0]) is True


def test_mpegts_audio_continuity_counter_advances() -> None:
    """Audio continuity counters should advance across payload calls."""
    # Given: A muxer that has already emitted one audio payload
    muxer = TapoPCMATransportStreamMuxer()
    first = _packets(muxer.audio_payload(b"\xaa" * 160))

    # When: Emitting a second payload
    second = _packets(muxer.audio_payload(b"\xbb" * 160))

    # Then: The continuity counter continues on the audio PID
    assert _continuity_counter(first[-1]) == 0
    assert _continuity_counter(second[0]) == 1


def test_mpegts_audio_pts_increments_by_pcma_sample_count() -> None:
    """Audio PTS should advance by the number of PCMA samples sent."""
    # Given: A muxer with deterministic initial timestamp
    muxer = TapoPCMATransportStreamMuxer()

    # When: Emitting two 160-sample PCMA frames
    first = _packets(muxer.audio_payload(b"\xaa" * 160))[0]
    second = _packets(muxer.audio_payload(b"\xbb" * 160))[0]

    # Then: The second PES timestamp advances by the first payload sample count
    assert _pts_from_first_packet(first) == 0
    assert _pts_from_first_packet(second) == 160


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
    assert len(data) % TS_PACKET_SIZE == 0
    return [data[index : index + TS_PACKET_SIZE] for index in range(0, len(data), TS_PACKET_SIZE)]


def _pid(packet: bytes) -> int:
    return ((packet[1] & 0x1F) << 8) | packet[2]


def _payload_unit_start(packet: bytes) -> bool:
    return bool(packet[1] & 0x40)


def _continuity_counter(packet: bytes) -> int:
    return packet[3] & 0x0F


def _payload(packet: bytes) -> bytes:
    adaptation_control = (packet[3] >> 4) & 0x03
    offset = 4
    if adaptation_control == 0x03:
        offset += 1 + packet[offset]
    payload = packet[offset:]
    if _payload_unit_start(packet) and _pid(packet) in {PAT_PID, PMT_PID}:
        pointer = payload[0]
        payload = payload[1 + pointer :]
    return payload


def _pts_from_first_packet(packet: bytes) -> int:
    payload = _payload(packet)
    assert payload[:4] == b"\x00\x00\x01\xc0"
    pts = payload[9:14]
    return (
        ((pts[0] >> 1) & 0x07) << 30
        | pts[1] << 22
        | ((pts[2] >> 1) & 0x7F) << 15
        | pts[3] << 7
        | ((pts[4] >> 1) & 0x7F)
    )
