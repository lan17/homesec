"""Small PCM resampling helpers for push-to-talk audio."""

from __future__ import annotations

import struct

_MIN_PCM16 = -32768
_MAX_PCM16 = 32767


def resample_pcm_s16le_mono(
    pcm_s16le: bytes,
    *,
    input_rate: int,
    output_rate: int,
) -> bytes:
    """Resample signed little-endian mono PCM without external media components.

    The browser sends 16 kHz mono PCM for the MVP. Most ONVIF backchannels speak
    G.711 at 8 kHz, so this function primarily down-samples 16 kHz -> 8 kHz.
    Integer downsampling uses box averaging; other ratios use linear
    interpolation to keep the implementation deterministic and dependency-free.
    """
    if input_rate <= 0 or output_rate <= 0:
        raise ValueError("sample rates must be positive")
    if len(pcm_s16le) % 2 != 0:
        raise ValueError("PCM S16LE input length must be even")
    if input_rate == output_rate or not pcm_s16le:
        return pcm_s16le

    samples = [sample for (sample,) in struct.iter_unpack("<h", pcm_s16le)]
    if input_rate > output_rate and input_rate % output_rate == 0:
        factor = input_rate // output_rate
        out = []
        for start in range(0, len(samples) - factor + 1, factor):
            out.append(_clamp(round(sum(samples[start : start + factor]) / factor)))
        return struct.pack(f"<{len(out)}h", *out) if out else b""

    output_count = int(round(len(samples) * output_rate / input_rate))
    if output_count <= 0:
        return b""
    out = []
    for index in range(output_count):
        pos = index * input_rate / output_rate
        left = int(pos)
        if left >= len(samples) - 1:
            out.append(samples[-1])
            continue
        frac = pos - left
        value = samples[left] * (1.0 - frac) + samples[left + 1] * frac
        out.append(_clamp(round(value)))
    return struct.pack(f"<{len(out)}h", *out)


def _clamp(value: int) -> int:
    return max(_MIN_PCM16, min(_MAX_PCM16, value))
