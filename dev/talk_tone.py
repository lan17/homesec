#!/usr/bin/env python3
"""Send a generated tone through an ONVIF RTSP audio backchannel.

Example:
    uv run python dev/talk_tone.py --rtsp-url rtsp://user:pass@camera/Streaming/Channels/101

This is a Phase 1 manual probe utility: it bypasses HomeSec runtime/API/UI and
exercises only the isolated RTSP/ONVIF backchannel protocol stack.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import struct
from collections.abc import Iterable

from homesec.sources.rtsp.talk.models import ONVIFBackchannelConfig
from homesec.sources.rtsp.talk.onvif_backchannel import ONVIFBackchannelAdapter
from homesec.sources.rtsp.talk.rtsp_auth import redact_rtsp_url

_INPUT_SAMPLE_RATE = 16_000
_FRAME_MS = 20


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a generated tone to an ONVIF RTSP backchannel"
    )
    url_group = parser.add_mutually_exclusive_group(required=True)
    url_group.add_argument("--rtsp-url", help="RTSP URL for the camera talk/backchannel endpoint")
    url_group.add_argument("--rtsp-url-env", help="Environment variable containing the RTSP URL")
    parser.add_argument("--username", help="Optional RTSP username; URL userinfo also works")
    parser.add_argument("--password", help="Optional RTSP password; URL userinfo also works")
    parser.add_argument("--duration-s", type=float, default=2.0, help="Tone duration in seconds")
    parser.add_argument("--frequency-hz", type=float, default=440.0, help="Tone frequency in Hz")
    parser.add_argument("--amplitude", type=float, default=0.20, help="Tone amplitude, 0.0..1.0")
    parser.add_argument(
        "--camera-name", default="manual-tone", help="Label used in local session metadata"
    )
    return parser


def _pcm_tone_frames(
    *,
    duration_s: float,
    frequency_hz: float,
    amplitude: float,
) -> Iterable[bytes]:
    if duration_s <= 0:
        raise ValueError("duration-s must be positive")
    if frequency_hz <= 0:
        raise ValueError("frequency-hz must be positive")
    if not 0.0 <= amplitude <= 1.0:
        raise ValueError("amplitude must be between 0.0 and 1.0")

    samples_per_frame = _INPUT_SAMPLE_RATE * _FRAME_MS // 1000
    frame_count = max(1, math.ceil(duration_s * 1000 / _FRAME_MS))
    max_value = int(32767 * amplitude)
    sample_index = 0
    for _ in range(frame_count):
        samples: list[int] = []
        for _sample in range(samples_per_frame):
            phase = 2.0 * math.pi * frequency_hz * sample_index / _INPUT_SAMPLE_RATE
            samples.append(int(max_value * math.sin(phase)))
            sample_index += 1
        yield struct.pack(f"<{len(samples)}h", *samples)


async def _run(args: argparse.Namespace) -> None:
    config = ONVIFBackchannelConfig(
        rtsp_url=args.rtsp_url,
        rtsp_url_env=args.rtsp_url_env,
        username=args.username,
        password=args.password,
    )
    adapter = ONVIFBackchannelAdapter(config, camera_name=args.camera_name)
    redacted_url = redact_rtsp_url(config.resolve_rtsp_url())
    print(f"Opening ONVIF RTSP backchannel to {redacted_url} ...")
    session = await adapter.open_session(
        session_id="manual-tone", input_sample_rate=_INPUT_SAMPLE_RATE
    )
    try:
        print(f"Backchannel ready: codec={session.selected_codec}; sending tone ...")
        for frame in _pcm_tone_frames(
            duration_s=args.duration_s,
            frequency_hz=args.frequency_hz,
            amplitude=args.amplitude,
        ):
            await session.write_pcm_frame(frame)
            await asyncio.sleep(_FRAME_MS / 1000)
    finally:
        await session.close()
    print("Tone complete; RTSP session closed.")


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
