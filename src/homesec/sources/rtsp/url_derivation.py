from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class DerivedRTSPUrl:
    url: str
    source: str


def derive_detect_rtsp_url(rtsp_url: str) -> DerivedRTSPUrl | None:
    """Derive a likely lower-cost detect stream URL from a primary URL."""
    subtype_derived = _replace_subtype_value(rtsp_url, from_value="0", to_value="1")
    if subtype_derived is not None:
        return DerivedRTSPUrl(url=subtype_derived, source="derived_subtype=1")

    stream_derived = _replace_stream_suffix(rtsp_url, from_value="1", to_value="2")
    if stream_derived is not None:
        return DerivedRTSPUrl(url=stream_derived, source="derived_stream2")

    return None


def derive_probe_candidate_urls(rtsp_url: str) -> list[str]:
    """Derive additional RTSP candidates worth probing during startup preflight."""
    candidates: list[str] = []
    for derived in (
        _replace_subtype_value(rtsp_url, from_value="0", to_value="1"),
        _replace_subtype_value(rtsp_url, from_value="1", to_value="0"),
        _replace_stream_suffix(rtsp_url, from_value="1", to_value="2"),
        _replace_stream_suffix(rtsp_url, from_value="2", to_value="1"),
    ):
        if derived is None or derived == rtsp_url:
            continue
        if derived in candidates:
            continue
        candidates.append(derived)
    return candidates


def _replace_subtype_value(rtsp_url: str, *, from_value: str, to_value: str) -> str | None:
    pattern = re.compile(rf"([?&]subtype=){re.escape(from_value)}(?=(&|$))", re.IGNORECASE)
    replaced = pattern.sub(rf"\g<1>{to_value}", rtsp_url, count=1)
    if replaced == rtsp_url:
        return None
    return replaced


def _replace_stream_suffix(rtsp_url: str, *, from_value: str, to_value: str) -> str | None:
    parsed = urlsplit(rtsp_url)
    path = parsed.path
    if not path:
        return None

    replacement_pattern = re.compile(rf"(?i)(/stream){re.escape(from_value)}(?=/?$)")
    replaced_path = replacement_pattern.sub(rf"\g<1>{to_value}", path, count=1)
    if replaced_path == path:
        return None

    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            replaced_path,
            parsed.query,
            parsed.fragment,
        )
    )
