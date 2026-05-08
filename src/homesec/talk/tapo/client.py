"""Async local HTTP/multipart client for the Tapo talk endpoint."""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field

from homesec.talk.backends import TalkBackendContext
from homesec.talk.tapo.config import (
    TapoCredential,
    TapoLocalTalkConfig,
    resolve_tapo_credential,
    resolve_tapo_host,
)
from homesec.talk.tapo.digest import (
    TapoDigestChallenge,
    TapoDigestError,
    build_digest_authorization_header,
    parse_www_authenticate,
)
from homesec.talk.tapo.multipart import (
    CLIENT_BOUNDARY,
    CLIENT_PART_PREFIX,
    TapoMultipartError,
    multipart_part,
    read_multipart_part,
)

_CRLF = "\r\n"
_HEADER_SEPARATOR = b"\r\n\r\n"
_STREAM_URI = "/stream"
_MAX_SETUP_RESPONSE_BYTES = 64 * 1024
_MAX_HTTP_RESPONSE_BODY_BYTES = 64 * 1024
_DEFAULT_CONNECT_TIMEOUT_S = 5.0
_DEFAULT_IO_TIMEOUT_S = 5.0


class TapoClientError(RuntimeError):
    """Base class for safe Tapo client failures."""


class TapoAuthError(TapoClientError):
    """Raised when the Tapo endpoint rejects authentication."""

    def __init__(self, message: str = "Tapo local authentication failed") -> None:
        super().__init__(message)


class TapoUnsupportedEndpointError(TapoClientError):
    """Raised when the local stream endpoint does not look like Tapo talk."""


class TapoProtocolError(TapoClientError):
    """Raised for malformed or rejected Tapo local protocol data."""


@dataclass(frozen=True, slots=True)
class TapoHTTPResponse:
    """Parsed HTTP response headers and optional body."""

    status_code: int
    reason: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    def header(self, name: str) -> str | None:
        """Return a case-insensitive response header."""
        return self.headers.get(name.lower())


@dataclass(slots=True)
class TapoLocalClient:
    """Connected Tapo local stream client after talk setup succeeds."""

    host: str
    port: int
    io_timeout_s: float
    reader: asyncio.StreamReader = field(repr=False)
    writer: asyncio.StreamWriter = field(repr=False)
    tapo_session_id: str = field(repr=False)
    _write_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    async def write_audio_mp2t(self, payload: bytes) -> None:
        """Write one audio/mp2t multipart chunk to the Tapo stream."""
        if self._closed:
            raise TapoProtocolError("Cannot write to closed Tapo local stream")
        if not payload:
            raise TapoProtocolError("Tapo audio payload must not be empty")
        part = multipart_part(
            CLIENT_PART_PREFIX,
            {
                "Content-Type": "audio/mp2t",
                "X-If-Encrypt": "0",
                "X-Session-Id": self.tapo_session_id,
            },
            payload,
        )
        async with self._write_lock:
            try:
                self.writer.write(part)
                await asyncio.wait_for(self.writer.drain(), timeout=self.io_timeout_s)
            except BaseException:
                self._closed = True
                await _close_writer_best_effort(self.writer, timeout_s=self.io_timeout_s)
                raise

    async def close(self) -> None:
        """Close the local stream connection."""
        if self._closed:
            return
        self._closed = True
        await _close_writer_best_effort(self.writer, timeout_s=self.io_timeout_s)


async def open_tapo_local_client(
    config: TapoLocalTalkConfig,
    context: TalkBackendContext,
) -> TapoLocalClient:
    """Authenticate to the local Tapo stream endpoint and run talk setup."""
    host = resolve_tapo_host(config, context)
    port = config.port
    connect_timeout_s = config.connect_timeout_s or context.source_connect_timeout_s
    io_timeout_s = config.io_timeout_s or context.source_io_timeout_s
    if connect_timeout_s <= 0:
        connect_timeout_s = _DEFAULT_CONNECT_TIMEOUT_S
    if io_timeout_s <= 0:
        io_timeout_s = _DEFAULT_IO_TIMEOUT_S

    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(host, port),
        timeout=connect_timeout_s,
    )
    try:
        challenge = await _request_digest_challenge(
            reader,
            writer,
            host=host,
            port=port,
            io_timeout_s=io_timeout_s,
        )
        credential = resolve_tapo_credential(
            config,
            context,
            preferred_hash_kind=challenge.preferred_hash_kind,
        )
        response_boundary = await _authenticate_stream(
            reader,
            writer,
            host=host,
            port=port,
            credential=credential,
            challenge=challenge,
            io_timeout_s=io_timeout_s,
        )
        tapo_session_id = await _setup_talk(
            reader,
            writer,
            response_boundary=response_boundary,
            mode=config.mode,
            io_timeout_s=io_timeout_s,
        )
        return TapoLocalClient(
            host=host,
            port=port,
            io_timeout_s=io_timeout_s,
            reader=reader,
            writer=writer,
            tapo_session_id=tapo_session_id,
        )
    except BaseException:
        await _close_writer_best_effort(writer, timeout_s=io_timeout_s)
        raise


async def _request_digest_challenge(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    host: str,
    port: int,
    io_timeout_s: float,
) -> TapoDigestChallenge:
    await _write_stream_request(
        writer,
        host=host,
        port=port,
        headers={},
        io_timeout_s=io_timeout_s,
    )
    response = await _read_http_response(reader, io_timeout_s=io_timeout_s, read_body=True)
    if response.status_code != 401:
        raise TapoUnsupportedEndpointError(
            "Tapo local endpoint did not request Digest authentication"
        )
    authenticate = response.header("www-authenticate")
    if authenticate is None:
        raise TapoUnsupportedEndpointError(
            "Tapo local endpoint did not provide Digest authentication"
        )
    challenge = parse_www_authenticate(authenticate)
    if challenge.scheme != "digest":
        raise TapoUnsupportedEndpointError(
            "Tapo local endpoint requires unsupported authentication"
        )
    return challenge


async def _authenticate_stream(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    host: str,
    port: int,
    credential: TapoCredential,
    challenge: TapoDigestChallenge,
    io_timeout_s: float,
) -> str:
    try:
        authorization = build_digest_authorization_header(
            challenge=challenge,
            method="POST",
            uri=_STREAM_URI,
            username=credential.username,
            password_material=credential.password_hash,
            nonce_count=1,
            cnonce=secrets.token_hex(8),
        )
    except TapoDigestError as exc:
        raise TapoProtocolError("Tapo local Digest challenge is not supported") from exc

    await _write_stream_request(
        writer,
        host=host,
        port=port,
        headers={"Authorization": authorization},
        io_timeout_s=io_timeout_s,
    )
    response = await _read_http_response(reader, io_timeout_s=io_timeout_s, read_body=False)
    if response.status_code == 401:
        raise TapoAuthError()
    if response.status_code != 200:
        raise TapoProtocolError("Tapo local endpoint returned an unexpected HTTP status")
    boundary = _boundary_from_content_type(response.header("content-type"))
    if boundary is None:
        raise TapoProtocolError("Tapo local endpoint did not provide a multipart boundary")
    return boundary


async def _setup_talk(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    response_boundary: str,
    mode: str,
    io_timeout_s: float,
) -> str:
    body = _talk_setup_body(mode=mode)
    part = multipart_part(
        CLIENT_PART_PREFIX,
        {"Content-Type": "application/json"},
        body,
    )
    writer.write(part)
    await asyncio.wait_for(writer.drain(), timeout=io_timeout_s)

    try:
        response = await read_multipart_part(
            reader,
            boundary=response_boundary,
            max_payload_bytes=_MAX_SETUP_RESPONSE_BYTES,
            timeout_s=io_timeout_s,
        )
    except TapoMultipartError as exc:
        raise TapoProtocolError("Tapo local talk setup multipart response failed") from exc
    if not _is_json_content_type(response.header("content-type")):
        raise TapoProtocolError("Tapo local talk setup returned a non-JSON response")
    try:
        payload = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TapoProtocolError("Tapo local talk setup returned malformed JSON") from exc
    session_id = payload.get("params", {}).get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise TapoProtocolError("Tapo local talk setup did not return a session id")
    return session_id.strip()


async def _write_stream_request(
    writer: asyncio.StreamWriter,
    *,
    host: str,
    port: int,
    headers: Mapping[str, str],
    io_timeout_s: float,
) -> None:
    merged: dict[str, str] = {
        "Host": f"{host}:{port}",
        "Content-Type": f"multipart/mixed; boundary={CLIENT_BOUNDARY}",
        "Connection": "keep-alive",
    }
    merged.update(headers)
    lines = [f"POST {_STREAM_URI} HTTP/1.1"]
    lines.extend(f"{name}: {value}" for name, value in merged.items())
    writer.write((_CRLF.join(lines) + _CRLF * 2).encode("iso-8859-1"))
    await asyncio.wait_for(writer.drain(), timeout=io_timeout_s)


async def _read_http_response(
    reader: asyncio.StreamReader,
    *,
    io_timeout_s: float,
    read_body: bool,
) -> TapoHTTPResponse:
    try:
        head = await asyncio.wait_for(reader.readuntil(_HEADER_SEPARATOR), timeout=io_timeout_s)
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError) as exc:
        raise TapoProtocolError("Malformed Tapo local HTTP response") from exc
    text = head.decode("iso-8859-1")
    lines = text.split(_CRLF)
    if not lines or not lines[0].startswith("HTTP/"):
        raise TapoProtocolError("Malformed Tapo local HTTP response")
    parts = lines[0].split(" ", maxsplit=2)
    if len(parts) < 2:
        raise TapoProtocolError("Malformed Tapo local HTTP status line")
    try:
        status_code = int(parts[1])
    except ValueError as exc:
        raise TapoProtocolError("Malformed Tapo local HTTP status line") from exc
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        name, separator, value = line.partition(":")
        if not separator:
            raise TapoProtocolError("Malformed Tapo local HTTP header")
        headers[name.strip().lower()] = value.strip()
    body = b""
    if read_body:
        content_length = _parse_content_length(headers.get("content-length"))
        if content_length > _MAX_HTTP_RESPONSE_BODY_BYTES:
            raise TapoProtocolError("Tapo local HTTP response body exceeds configured limit")
        if content_length:
            try:
                body = await asyncio.wait_for(
                    reader.readexactly(content_length),
                    timeout=io_timeout_s,
                )
            except asyncio.IncompleteReadError as exc:
                raise TapoProtocolError("Malformed Tapo local HTTP response body") from exc
    return TapoHTTPResponse(
        status_code=status_code,
        reason=parts[2].strip() if len(parts) > 2 else "",
        headers=headers,
        body=body,
    )


def _talk_setup_body(*, mode: str) -> bytes:
    return json.dumps(
        {
            "params": {"talk": {"mode": mode}, "method": "get"},
            "seq": 3,
            "type": "request",
        },
        separators=(",", ":"),
    ).encode("utf-8")


def _boundary_from_content_type(content_type: str | None) -> str | None:
    if content_type is None:
        return None
    for part in content_type.split(";"):
        name, separator, value = part.strip().partition("=")
        if separator and name.lower() == "boundary":
            return value.strip().strip('"')
    return None


def _is_json_content_type(content_type: str | None) -> bool:
    if content_type is None:
        return False
    media_type = content_type.split(";", maxsplit=1)[0].strip().lower()
    return media_type == "application/json"


def _parse_content_length(value: str | None) -> int:
    if value is None or value.strip() == "":
        return 0
    try:
        content_length = int(value)
    except ValueError as exc:
        raise TapoProtocolError("Invalid Tapo local HTTP Content-Length") from exc
    if content_length < 0:
        raise TapoProtocolError("Invalid Tapo local HTTP Content-Length")
    return content_length


async def _close_writer_best_effort(
    writer: asyncio.StreamWriter,
    *,
    timeout_s: float,
) -> None:
    writer.close()
    try:
        await asyncio.wait_for(writer.wait_closed(), timeout=max(timeout_s, 0.1))
    except asyncio.CancelledError:
        raise
    except Exception:
        pass
