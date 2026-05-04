"""Minimal SDP parser for ONVIF RTSP audio backchannel negotiation."""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

from homesec.sources.rtsp.talk.errors import (
    CameraBackchannelUnsupportedError,
    UnsupportedTalkCodecError,
)

_DIRECTION_ATTRIBUTES = {"sendonly", "recvonly", "sendrecv", "inactive"}
_STATIC_RTPMAP: dict[int, tuple[str, int, int]] = {
    0: ("PCMU", 8000, 1),
    8: ("PCMA", 8000, 1),
}


@dataclass(slots=True, frozen=True)
class SDPCodec:
    """RTP codec advertised in an SDP media description."""

    payload_type: int
    encoding_name: str
    clock_rate: int
    channels: int = 1
    fmtp: str | None = None

    @property
    def normalized_name(self) -> str:
        base = f"{self.encoding_name.upper()}/{self.clock_rate}"
        if self.channels != 1:
            return f"{base}/{self.channels}"
        return base


@dataclass(slots=True)
class SDPMediaDescription:
    """One SDP m= section."""

    media: str
    port: int
    proto: str
    payload_types: list[int]
    direction: str | None = None
    control: str | None = None
    codecs: dict[int, SDPCodec] = field(default_factory=dict)
    attributes: dict[str, list[str | None]] = field(default_factory=dict)

    def codec_for_payload(self, payload_type: int) -> SDPCodec | None:
        codec = self.codecs.get(payload_type)
        if codec is not None:
            return codec
        static = _STATIC_RTPMAP.get(payload_type)
        if static is None:
            return None
        encoding, clock_rate, channels = static
        return SDPCodec(
            payload_type=payload_type,
            encoding_name=encoding,
            clock_rate=clock_rate,
            channels=channels,
        )


@dataclass(slots=True)
class SDPDescription:
    """Parsed SDP session with media sections."""

    session_direction: str | None = None
    session_control: str | None = None
    media: list[SDPMediaDescription] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class SelectedBackchannel:
    """Selected audio backchannel stream and codec."""

    control: str
    payload_type: int
    codec: SDPCodec
    media: SDPMediaDescription

    @property
    def selected_codec(self) -> str:
        return self.codec.normalized_name


def parse_sdp(sdp: str) -> SDPDescription:
    """Parse enough SDP to discover ONVIF sendonly audio backchannel streams."""
    description = SDPDescription()
    current: SDPMediaDescription | None = None

    for raw_line in sdp.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line or len(line) < 2 or line[1] != "=":
            continue
        prefix = line[0]
        value = line[2:].strip()

        if prefix == "m":
            parts = value.split()
            if len(parts) < 4:
                continue
            try:
                port = int(parts[1])
                payload_types = [int(item) for item in parts[3:]]
            except ValueError:
                continue
            current = SDPMediaDescription(
                media=parts[0].lower(),
                port=port,
                proto=parts[2],
                payload_types=payload_types,
            )
            description.media.append(current)
            continue

        if prefix != "a":
            continue

        name, separator, attr_value = value.partition(":")
        name = name.strip().lower()
        attr = attr_value.strip() if separator else None

        if current is None:
            if name in _DIRECTION_ATTRIBUTES:
                description.session_direction = name
            elif name == "control" and attr:
                description.session_control = attr
            continue

        current.attributes.setdefault(name, []).append(attr)
        if name in _DIRECTION_ATTRIBUTES:
            current.direction = name
        elif name == "control" and attr:
            current.control = attr
        elif name == "rtpmap" and attr:
            codec = _parse_rtpmap(attr)
            if codec is not None:
                current.codecs[codec.payload_type] = codec
        elif name == "fmtp" and attr:
            _apply_fmtp(current, attr)

    return description


def select_audio_backchannel(
    sdp: str | SDPDescription,
    *,
    preferred_codecs: list[str],
    base_control_url: str | None = None,
) -> SelectedBackchannel:
    """Select the first sendonly audio stream matching the preferred codecs.

    Raises specific errors so the runtime/source layer can return a stable status
    without leaking raw camera protocol details to API callers.
    """
    description = parse_sdp(sdp) if isinstance(sdp, str) else sdp
    preferences = [_normalize_codec_name(item) for item in preferred_codecs]
    candidates: list[SDPMediaDescription] = []

    for media in description.media:
        direction = media.direction or description.session_direction
        if media.media == "audio" and direction == "sendonly":
            candidates.append(media)

    if not candidates:
        raise CameraBackchannelUnsupportedError("SDP has no sendonly audio backchannel")

    for preference in preferences:
        for media in candidates:
            for payload_type in media.payload_types:
                codec = media.codec_for_payload(payload_type)
                if codec is None:
                    continue
                if _normalize_codec_name(codec.normalized_name) != preference:
                    continue
                control = _resolve_control_url(
                    description=description,
                    media=media,
                    base_control_url=base_control_url,
                )
                return SelectedBackchannel(
                    control=control,
                    payload_type=payload_type,
                    codec=codec,
                    media=media,
                )

    raise UnsupportedTalkCodecError("SDP sendonly audio has no preferred codec")


def _parse_rtpmap(attr: str) -> SDPCodec | None:
    payload_text, separator, encoding_text = attr.partition(" ")
    if not separator:
        return None
    try:
        payload_type = int(payload_text)
    except ValueError:
        return None
    codec_parts = encoding_text.strip().split("/")
    if len(codec_parts) < 2:
        return None
    try:
        clock_rate = int(codec_parts[1])
        channels = int(codec_parts[2]) if len(codec_parts) >= 3 else 1
    except ValueError:
        return None
    return SDPCodec(
        payload_type=payload_type,
        encoding_name=codec_parts[0].upper(),
        clock_rate=clock_rate,
        channels=channels,
    )


def _apply_fmtp(media: SDPMediaDescription, attr: str) -> None:
    payload_text, separator, fmtp = attr.partition(" ")
    if not separator:
        return
    try:
        payload_type = int(payload_text)
    except ValueError:
        return
    codec = media.codecs.get(payload_type)
    if codec is None:
        return
    media.codecs[payload_type] = SDPCodec(
        payload_type=codec.payload_type,
        encoding_name=codec.encoding_name,
        clock_rate=codec.clock_rate,
        channels=codec.channels,
        fmtp=fmtp.strip(),
    )


def _resolve_control_url(
    *,
    description: SDPDescription,
    media: SDPMediaDescription,
    base_control_url: str | None,
) -> str:
    control = media.control or description.session_control or "*"
    if base_control_url is None:
        return control
    if media.control is not None and description.session_control:
        session_base = _join_control_url(base_control_url, description.session_control)
        return _join_control_url(session_base, media.control)
    return _join_control_url(base_control_url, control)


def _normalize_codec_name(value: str) -> str:
    parts = value.strip().split("/")
    if len(parts) < 2:
        return value.strip().upper()
    encoding = parts[0].upper()
    clock = parts[1]
    channels = parts[2] if len(parts) >= 3 else None
    if channels in (None, "", "1"):
        return f"{encoding}/{clock}"
    return f"{encoding}/{clock}/{channels}"


def _join_control_url(base_url: str, control: str) -> str:
    control = control.strip()
    if "://" in control:
        return control
    if control == "*":
        return base_url

    base = urlsplit(base_url)
    control_parts = urlsplit(control)
    if control.startswith("/"):
        return urlunsplit(
            (
                base.scheme,
                base.netloc,
                control_parts.path,
                control_parts.query,
                control_parts.fragment,
            )
        )

    base_path = base.path or ""
    if not base_path:
        joined_path = f"/{control_parts.path}"
    elif base_path.endswith("/"):
        joined_path = f"{base_path}{control_parts.path}"
    else:
        joined_path = f"{base_path}/{control_parts.path}"
    return urlunsplit(
        (
            base.scheme,
            base.netloc,
            joined_path,
            control_parts.query,
            control_parts.fragment,
        )
    )
