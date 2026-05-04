from __future__ import annotations

import struct

import pytest

from homesec.sources.rtsp.talk.g711 import encode_pcmu
from homesec.sources.rtsp.talk.resample import resample_pcm_s16le_mono
from homesec.sources.rtsp.talk.rtp import RTPPacketizer, parse_rtp_header


def _pcm(*samples: int) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def test_pcmu_encode_known_vectors() -> None:
    samples = (-32768, -32124, -1, 0, 1, 32124, 32767)

    assert encode_pcmu(_pcm(*samples)) == bytes([0x00, 0x00, 0x7E, 0xFF, 0xFF, 0x80, 0x80])


def test_g711_rejects_odd_pcm_input() -> None:
    with pytest.raises(ValueError, match="even"):
        encode_pcmu(b"\x00")


def test_resample_16khz_to_8khz_uses_box_averaging() -> None:
    resampled = resample_pcm_s16le_mono(
        _pcm(0, 2, 10, 12, -4, -2, 100, 104),
        input_rate=16000,
        output_rate=8000,
    )

    assert struct.unpack("<4h", resampled) == (1, 11, -3, 102)


def test_rtp_packetizer_header_fields_and_timestamp_increments() -> None:
    packetizer = RTPPacketizer(
        payload_type=0,
        sequence_number=0xFFFF,
        timestamp=1000,
        ssrc=0x01020304,
    )

    packet = packetizer.packetize(b"\xff" * 160, timestamp_increment=160, marker=True)

    header = parse_rtp_header(packet)
    assert header == {
        "version": 2,
        "padding": False,
        "extension": False,
        "csrc_count": 0,
        "marker": True,
        "payload_type": 0,
        "sequence_number": 0xFFFF,
        "timestamp": 1000,
        "ssrc": 0x01020304,
    }
    assert packet[12:] == b"\xff" * 160
    assert packetizer.sequence_number == 0
    assert packetizer.timestamp == 1160

    next_packet = packetizer.packetize(b"\x7f" * 160, timestamp_increment=160)
    assert parse_rtp_header(next_packet)["sequence_number"] == 0
    assert parse_rtp_header(next_packet)["timestamp"] == 1160
