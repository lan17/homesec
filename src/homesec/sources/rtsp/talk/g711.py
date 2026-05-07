"""G.711 audio codecs used by ONVIF RTSP audio backchannels."""

from __future__ import annotations

import struct

_BIAS = 0x84
_CLIP = 32635
_SEG_END = (0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF)
_ALAW_MASK = 0xD5
_ALAW_SIGN_MASK = 0x80
_ALAW_QUANT_MASK = 0x0F
_ALAW_SEG_MASK = 0x70
_ALAW_SEG_SHIFT = 4


def encode_pcmu(pcm_s16le: bytes) -> bytes:
    """Encode little-endian signed 16-bit mono PCM to G.711 μ-law/PCMU bytes.

    HomeSec's browser contract sends deterministic PCM S16LE frames. ONVIF
    camera speakers commonly advertise PCMU/8000; each encoded byte represents
    one 8 kHz sample.
    """
    return bytes(_linear_to_ulaw(sample) for sample in _iter_pcm16le(pcm_s16le))


def encode_pcma(pcm_s16le: bytes) -> bytes:
    """Encode little-endian signed 16-bit mono PCM to G.711 A-law/PCMA bytes."""
    return bytes(_ALAW_ENCODE_TABLE[(sample + 32768) >> 2] for sample in _iter_pcm16le(pcm_s16le))


def _iter_pcm16le(pcm_s16le: bytes):  # type: ignore[no-untyped-def]
    if len(pcm_s16le) % 2 != 0:
        raise ValueError("PCM S16LE input length must be even")
    for (sample,) in struct.iter_unpack("<h", pcm_s16le):
        yield int(sample)


def _linear_to_ulaw(sample: int) -> int:
    """Convert one signed 16-bit PCM sample to ITU-T G.711 μ-law."""
    if sample < 0:
        # Match CPython audioop/standard lookup rounding: negative values are
        # offset so -1..-8 map to the first negative code below zero.
        pcm = _BIAS - sample + 3
        mask = 0x7F
    else:
        pcm = sample + _BIAS
        mask = 0xFF

    if pcm > _CLIP + _BIAS:
        pcm = _CLIP + _BIAS

    segment = _search_segment(pcm)
    if segment >= 8:
        return 0x7F ^ mask
    value = (segment << 4) | ((pcm >> (segment + 3)) & 0x0F)
    return value ^ mask


def _search_segment(value: int) -> int:
    for index, end in enumerate(_SEG_END):
        if value <= end:
            return index
    return len(_SEG_END)


def _build_alaw_encode_table() -> bytes:
    """Build a compact 16-bit PCM to A-law table matching standard G.711 quantization."""
    table = bytearray(16384)
    index = 1
    table[8192] = _ALAW_MASK

    for code in range(127):
        lower = _alaw_to_linear(code ^ _ALAW_MASK)
        upper = _alaw_to_linear((code + 1) ^ _ALAW_MASK)
        threshold = (lower + upper + 4) >> 3
        while index < threshold:
            table[8192 - index] = code ^ (_ALAW_MASK ^ _ALAW_SIGN_MASK)
            table[8192 + index] = code ^ _ALAW_MASK
            index += 1

    while index < 8192:
        table[8192 - index] = 127 ^ (_ALAW_MASK ^ _ALAW_SIGN_MASK)
        table[8192 + index] = 127 ^ _ALAW_MASK
        index += 1

    table[0] = table[1]
    return bytes(table)


def _alaw_to_linear(value: int) -> int:
    encoded = value ^ 0x55
    magnitude = encoded & _ALAW_QUANT_MASK
    segment = (encoded & _ALAW_SEG_MASK) >> _ALAW_SEG_SHIFT
    if segment:
        magnitude = (magnitude + magnitude + 1 + 32) << (segment + 2)
    else:
        magnitude = (magnitude + magnitude + 1) << 3
    return magnitude if encoded & _ALAW_SIGN_MASK else -magnitude


_ALAW_ENCODE_TABLE = _build_alaw_encode_table()
