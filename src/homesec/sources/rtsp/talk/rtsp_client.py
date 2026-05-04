"""Small async RTSP client primitives for ONVIF audio backchannel sessions."""

from __future__ import annotations

import asyncio
import re
import secrets
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from homesec.sources.rtsp.talk.errors import RTSPAuthenticationError, RTSPProtocolError
from homesec.sources.rtsp.talk.rtsp_auth import (
    RTSPAuthChallenge,
    RTSPCredentials,
    build_authorization_header,
    parse_www_authenticate,
    request_uri_without_credentials,
    split_rtsp_url_credentials,
)

_CRLF = "\r\n"
_HEADER_SEPARATOR = b"\r\n\r\n"


@dataclass(slots=True, frozen=True)
class RTSPResponse:
    """Parsed RTSP response."""

    status_code: int
    reason: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    @classmethod
    def parse(cls, raw: bytes) -> RTSPResponse:
        """Parse a full RTSP response byte string."""
        head, separator, body = raw.partition(_HEADER_SEPARATOR)
        if not separator:
            raise RTSPProtocolError("RTSP response missing header terminator")
        lines = head.decode("iso-8859-1").split(_CRLF)
        if not lines:
            raise RTSPProtocolError("Empty RTSP response")
        match = re.match(r"RTSP/\d\.\d\s+(\d{3})(?:\s+(.*))?$", lines[0])
        if match is None:
            raise RTSPProtocolError(f"Invalid RTSP status line: {lines[0]!r}")
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if not line:
                continue
            name, sep, value = line.partition(":")
            if not sep:
                raise RTSPProtocolError(f"Invalid RTSP header line: {line!r}")
            key = name.strip().lower()
            if key in headers:
                headers[key] = f"{headers[key]}, {value.strip()}"
            else:
                headers[key] = value.strip()
        content_length = int(headers.get("content-length", "0") or "0")
        if content_length and len(body) < content_length:
            raise RTSPProtocolError("RTSP response body shorter than Content-Length")
        if content_length:
            body = body[:content_length]
        return cls(
            status_code=int(match.group(1)),
            reason=(match.group(2) or "").strip(),
            headers=headers,
            body=body,
        )

    def header(self, name: str) -> str | None:
        """Return a case-insensitive header value."""
        return self.headers.get(name.lower())


@dataclass(slots=True)
class RTSPConnectionConfig:
    """Connection configuration for the minimal RTSP client."""

    url: str
    credentials: RTSPCredentials | None = None
    user_agent: str = "HomeSec/PushToTalk"
    connect_timeout_s: float = 5.0
    io_timeout_s: float = 5.0


class RTSPClient:
    """Async RTSP control client with Basic/Digest retry support."""

    def __init__(self, config: RTSPConnectionConfig) -> None:
        clean_url, url_credentials = split_rtsp_url_credentials(config.url)
        self._url = clean_url
        self._credentials = config.credentials or url_credentials
        self._user_agent = config.user_agent
        self._connect_timeout_s = config.connect_timeout_s
        self._io_timeout_s = config.io_timeout_s
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._cseq = 1
        self.session_id: str | None = None
        self._auth_challenge: RTSPAuthChallenge | None = None
        self._auth_nonce_count = 0
        self._auth_cnonce = ""

    async def connect(self) -> None:
        """Open the RTSP TCP connection."""
        parsed = urlsplit(self._url)
        if parsed.scheme.lower() != "rtsp":
            raise ValueError("RTSP URL must use rtsp://")
        host = parsed.hostname
        if host is None:
            raise ValueError("RTSP URL must include a host")
        port = parsed.port or 554
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=self._connect_timeout_s,
        )

    async def close(self) -> None:
        """Close the RTSP TCP connection."""
        writer = self._writer
        self._reader = None
        self._writer = None
        if writer is None:
            return
        writer.close()
        await writer.wait_closed()

    async def request(
        self,
        method: str,
        uri: str | None = None,
        *,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
        retry_auth: bool = True,
    ) -> RTSPResponse:
        """Send an RTSP request and return the parsed response."""
        request_uri = request_uri_without_credentials(uri or self._url)
        response = await self._request_once(method, request_uri, headers=headers, body=body)
        if response.status_code != 401 or not retry_auth:
            return response
        www_authenticate = response.header("www-authenticate")
        if www_authenticate is None or self._credentials is None:
            raise RTSPAuthenticationError(
                "Camera requested RTSP authentication but no credentials are configured"
            )
        self._auth_challenge = parse_www_authenticate(www_authenticate)
        self._auth_nonce_count = 0
        self._auth_cnonce = secrets.token_hex(8)
        retry_response = await self._request_once(method, request_uri, headers=headers, body=body)
        if retry_response.status_code == 401:
            raise RTSPAuthenticationError("RTSP authentication was rejected by the camera")
        return retry_response

    async def describe(self, *, headers: dict[str, str] | None = None) -> RTSPResponse:
        """Send DESCRIBE for the aggregate RTSP URL."""
        merged = {"Accept": "application/sdp"}
        if headers:
            merged.update(headers)
        return await self.request("DESCRIBE", self._url, headers=merged)

    async def setup_interleaved(
        self,
        control_url: str,
        *,
        rtp_channel: int = 0,
        headers: dict[str, str] | None = None,
    ) -> RTSPResponse:
        """SETUP a TCP-interleaved RTP channel."""
        merged = {"Transport": f"RTP/AVP/TCP;unicast;interleaved={rtp_channel}-{rtp_channel + 1}"}
        if headers:
            merged.update(headers)
        response = await self.request("SETUP", control_url, headers=merged)
        if response.status_code == 200:
            session_header = response.header("session")
            if session_header:
                self.session_id = session_header.split(";", maxsplit=1)[0].strip()
        return response

    async def play(self, *, headers: dict[str, str] | None = None) -> RTSPResponse:
        """Send PLAY for the aggregate RTSP URL."""
        return await self.request("PLAY", self._url, headers=headers)

    async def teardown(self, *, headers: dict[str, str] | None = None) -> RTSPResponse:
        """Send TEARDOWN for the aggregate RTSP URL."""
        return await self.request("TEARDOWN", self._url, headers=headers, retry_auth=False)

    async def send_interleaved_frame(self, channel: int, payload: bytes) -> None:
        """Send one RTSP TCP-interleaved binary frame."""
        if not 0 <= channel <= 255:
            raise ValueError("interleaved channel must be in range 0..255")
        if len(payload) > 0xFFFF:
            raise ValueError("interleaved payload exceeds 65535 bytes")
        writer = self._require_writer()
        writer.write(b"$" + bytes([channel]) + len(payload).to_bytes(2, "big") + payload)
        await asyncio.wait_for(writer.drain(), timeout=self._io_timeout_s)

    async def _request_once(
        self,
        method: str,
        uri: str,
        *,
        headers: dict[str, str] | None,
        body: bytes,
    ) -> RTSPResponse:
        writer = self._require_writer()
        request_headers = self._build_headers(method, uri, headers=headers, body=body)
        lines = [f"{method} {uri} RTSP/1.0"]
        lines.extend(f"{name}: {value}" for name, value in request_headers.items())
        raw = (_CRLF.join(lines) + _CRLF * 2).encode("iso-8859-1") + body
        writer.write(raw)
        await asyncio.wait_for(writer.drain(), timeout=self._io_timeout_s)
        return await self._read_response()

    def _build_headers(
        self,
        method: str,
        uri: str,
        *,
        headers: dict[str, str] | None,
        body: bytes,
    ) -> dict[str, str]:
        result: dict[str, str] = {
            "CSeq": str(self._cseq),
            "User-Agent": self._user_agent,
        }
        self._cseq += 1
        if self.session_id is not None and method.upper() not in {"DESCRIBE", "SETUP"}:
            result["Session"] = self.session_id
        if body:
            result["Content-Length"] = str(len(body))
        if headers:
            result.update(headers)
        if self._auth_challenge is not None and self._credentials is not None:
            self._auth_nonce_count += 1
            result["Authorization"] = build_authorization_header(
                challenge=self._auth_challenge,
                method=method,
                uri=uri,
                credentials=self._credentials,
                nonce_count=self._auth_nonce_count,
                cnonce=self._auth_cnonce,
            )
        return result

    async def _read_response(self) -> RTSPResponse:
        reader = self._require_reader()
        while True:
            first = await asyncio.wait_for(reader.readexactly(1), timeout=self._io_timeout_s)
            if first == b"$":
                await self._read_interleaved_frame(reader)
                continue

            head = first + await asyncio.wait_for(
                reader.readuntil(_HEADER_SEPARATOR), timeout=self._io_timeout_s
            )
            header_text = head.decode("iso-8859-1")
            content_length = _content_length_from_header_text(header_text)
            body = b""
            if content_length:
                body = await asyncio.wait_for(
                    reader.readexactly(content_length), timeout=self._io_timeout_s
                )
            return RTSPResponse.parse(head + body)

    async def _read_interleaved_frame(self, reader: asyncio.StreamReader) -> bytes:
        _channel = await asyncio.wait_for(reader.readexactly(1), timeout=self._io_timeout_s)
        length_bytes = await asyncio.wait_for(reader.readexactly(2), timeout=self._io_timeout_s)
        length = int.from_bytes(length_bytes, "big")
        return await asyncio.wait_for(reader.readexactly(length), timeout=self._io_timeout_s)

    def _require_reader(self) -> asyncio.StreamReader:
        if self._reader is None:
            raise RuntimeError("RTSP client is not connected")
        return self._reader

    def _require_writer(self) -> asyncio.StreamWriter:
        if self._writer is None:
            raise RuntimeError("RTSP client is not connected")
        return self._writer


def parse_interleaved_channels(transport_header: str) -> tuple[int, int] | None:
    """Parse interleaved RTP/RTCP channels from an RTSP Transport header."""
    match = re.search(r"(?:^|;)\s*interleaved\s*=\s*(\d+)\s*-\s*(\d+)", transport_header, re.I)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _content_length_from_header_text(header_text: str) -> int:
    for line in header_text.split(_CRLF):
        name, sep, value = line.partition(":")
        if sep and name.lower() == "content-length":
            return int(value.strip() or "0")
    return 0
