from __future__ import annotations

import struct

import pytest

from homesec.sources.rtsp.talk.g711 import encode_pcma, encode_pcmu
from homesec.sources.rtsp.talk.resample import resample_pcm_s16le_mono
from homesec.sources.rtsp.talk.rtp import RTPPacketizer, parse_rtp_header


def _pcm(*samples: int) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def test_pcmu_encode_known_vectors() -> None:
    # Given: Signed 16-bit PCM samples with known G.711 u-law reference values.
    # When: The samples are encoded as PCMU.
    # Then: The encoder returns the expected byte vector.
    samples = (-32768, -32124, -1, 0, 1, 32124, 32767)

    assert encode_pcmu(_pcm(*samples)) == bytes([0x00, 0x00, 0x7E, 0xFF, 0xFF, 0x80, 0x80])


def test_pcmu_encode_clips_out_of_range_extremes() -> None:
    # Given: PCM samples at the signed 16-bit extremes.
    # When: The samples are encoded as PCMU.
    # Then: The encoder clips them to the expected u-law codewords.
    assert encode_pcmu(_pcm(-32768, 32767)) == bytes([0x00, 0x80])


def test_pcma_encode_known_vectors() -> None:
    # Given: Representative signed 16-bit PCM samples spanning the A-law range.
    samples = (-32768, -32124, -1, 0, 1, 32124, 32767)

    # When: The samples are encoded as G.711 A-law.
    encoded = encode_pcma(_pcm(*samples))

    # Then: The bytes match the standard PCMA quantization points.
    assert encoded == bytes([0x2A, 0x2A, 0x55, 0xD5, 0xD5, 0xAA, 0xAA])


def test_g711_rejects_odd_pcm_input() -> None:
    # Given: G.711 encoders receive a byte string that is not whole 16-bit PCM samples.
    # When: PCMU or PCMA encoding is requested.
    # Then: The encoders reject the odd-length input.
    with pytest.raises(ValueError, match="even"):
        encode_pcmu(b"\x00")


def test_resample_16khz_to_8khz_uses_box_averaging() -> None:
    # Given: 16 kHz mono PCM samples that downsample cleanly by a factor of two.
    # When: The audio is resampled to 8 kHz.
    # Then: Each output sample is the average of the corresponding input pair.
    resampled = resample_pcm_s16le_mono(
        _pcm(0, 2, 10, 12, -4, -2, 100, 104),
        input_rate=16000,
        output_rate=8000,
    )

    assert struct.unpack("<4h", resampled) == (1, 11, -3, 102)


def test_resample_same_rate_and_empty_input_are_passthrough() -> None:
    # Given: PCM input that either keeps the same rate or contains no samples.
    # When: The resampler is asked to produce the same target rate.
    # Then: It returns the original bytes or empty bytes unchanged.
    pcm = _pcm(1, -2, 3)

    assert resample_pcm_s16le_mono(pcm, input_rate=8000, output_rate=8000) == pcm
    assert resample_pcm_s16le_mono(b"", input_rate=16000, output_rate=8000) == b""


def test_resample_rejects_invalid_rates_and_odd_input() -> None:
    # Given: Resampling input with invalid rates or incomplete PCM samples.
    # When: Resampling is requested.
    # Then: The resampler rejects each invalid contract.
    with pytest.raises(ValueError, match="positive"):
        resample_pcm_s16le_mono(_pcm(1), input_rate=0, output_rate=8000)

    with pytest.raises(ValueError, match="positive"):
        resample_pcm_s16le_mono(_pcm(1), input_rate=16000, output_rate=-1)

    with pytest.raises(ValueError, match="even"):
        resample_pcm_s16le_mono(b"\x00", input_rate=16000, output_rate=8000)


def test_resample_non_integer_ratio_uses_linear_interpolation() -> None:
    # Given: PCM samples whose input and output rates have a non-integer ratio.
    # When: The audio is resampled.
    # Then: The output uses linear interpolation at the target sample positions.
    resampled = resample_pcm_s16le_mono(
        _pcm(0, 100, 200),
        input_rate=2,
        output_rate=3,
    )

    assert struct.unpack("<4h", resampled) == (0, 67, 133, 200)


def test_resample_returns_empty_when_integer_downsample_has_too_few_samples() -> None:
    # Given: Integer downsampling receives fewer input samples than one output window.
    # When: The input is resampled from 16 kHz to 8 kHz.
    # Then: The resampler returns empty audio rather than a partial sample.
    assert resample_pcm_s16le_mono(_pcm(10), input_rate=16000, output_rate=8000) == b""


def test_rtp_packetizer_header_fields_and_timestamp_increments() -> None:
    # Given: An RTP packetizer with fixed sequence, timestamp, and SSRC seeds.
    # When: Two payloads are packetized with explicit timestamp increments.
    # Then: The RTP headers and timestamp progression match the talk stream contract.
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


def test_rtp_packetizer_defaults_timestamp_increment_to_payload_length() -> None:
    # Given: An RTP packetizer sends an audio payload without an explicit increment.
    # When: The payload is packetized.
    # Then: The packetizer advances the timestamp by payload length.
    packetizer = RTPPacketizer(
        payload_type=0,
        sequence_number=1,
        timestamp=10,
        ssrc=1,
    )

    packetizer.packetize(b"12345")

    assert packetizer.timestamp == 15


def test_rtp_packetizer_validates_configuration_and_increment() -> None:
    # Given: RTP packetizer construction and packet calls receive invalid values.
    # When: The invalid payload type, sequence, SSRC, or increment is used.
    # Then: The packetizer rejects those values before emitting packets.
    with pytest.raises(ValueError, match="payload type"):
        RTPPacketizer(payload_type=128)

    with pytest.raises(ValueError, match="clock_rate"):
        RTPPacketizer(payload_type=0, clock_rate=0)

    with pytest.raises(ValueError, match="timestamp_increment"):
        RTPPacketizer(payload_type=0).packetize(b"payload", timestamp_increment=-1)


def test_parse_rtp_header_rejects_short_packets() -> None:
    # Given: A byte string shorter than the fixed RTP header.
    # When: The RTP header parser reads it.
    # Then: The parser rejects the truncated packet.
    with pytest.raises(ValueError, match="shorter"):
        parse_rtp_header(b"\x80")
