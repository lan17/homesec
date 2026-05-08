"""Fake Tapo local stream endpoint for protocol tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Literal

from homesec.talk.tapo.digest import (
    TapoDigestChallenge,
    digest_authorization_matches,
    parse_www_authenticate,
)
from homesec.talk.tapo.multipart import (
    CLIENT_BOUNDARY,
    DEVICE_BOUNDARY,
    DEVICE_PART_PREFIX,
    TapoMultipartPart,
    multipart_part,
    read_multipart_part,
)

_HEADER_SEPARATOR = b"\r\n\r\n"


@dataclass(frozen=True, slots=True)
class FakeTapoHTTPRequest:
    """HTTP request captured by the fake Tapo endpoint."""

    method: str
    uri: str
    headers: dict[str, str]


@dataclass(slots=True)
class FakeTapoServer:
    """Minimal local Tapo endpoint with Digest auth and multipart talk setup."""

    hash_kind: Literal["sha256", "md5"] = "sha256"
    username: str = "admin"
    credential_hash: str = "A" * 64
    session_id: str = "tapo-session-1"
    reject_auth: bool = False
    omit_session_id: bool = False
    malformed_setup_json: bool = False
    setup_content_type: str = "application/json"
    challenge_qop: str | None = "auth"
    requests: list[FakeTapoHTTPRequest] = field(default_factory=list)
    setup_parts: list[TapoMultipartPart] = field(default_factory=list)
    audio_parts: list[TapoMultipartPart] = field(default_factory=list)
    closed_connections: int = 0
    _server: asyncio.AbstractServer | None = field(default=None, init=False, repr=False)
    _port: int | None = field(default=None, init=False, repr=False)
    _tasks: set[asyncio.Task[None]] = field(default_factory=set, init=False, repr=False)

    @property
    def host(self) -> str:
        """Return the bound host."""
        return "127.0.0.1"

    @property
    def port(self) -> int:
        """Return the bound port."""
        if self._port is None:
            raise RuntimeError("fake Tapo server is not started")
        return self._port

    async def start(self) -> None:
        """Start the fake endpoint."""
        self._server = await asyncio.start_server(self._handle_client, self.host, 0)
        sockets = self._server.sockets or ()
        if not sockets:
            raise RuntimeError("fake Tapo server did not bind a socket")
        self._port = int(sockets[0].getsockname()[1])

    async def stop(self) -> None:
        """Stop the fake endpoint and wait for client handlers."""
        server = self._server
        self._server = None
        self._port = None
        if server is not None:
            server.close()
            await server.wait_closed()
        tasks = list(self._tasks)
        if tasks:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        try:
            request = await self._read_request(reader)
            if not self._authorization_matches(request):
                writer.write(
                    _http_response(
                        status=401,
                        reason="Unauthorized",
                        headers={"WWW-Authenticate": self._challenge_header()},
                    )
                )
                await writer.drain()

            request = await self._read_request(reader)
            if not self._authorization_matches(request) or self.reject_auth:
                writer.write(
                    _http_response(
                        status=401,
                        reason="Unauthorized",
                        headers={"WWW-Authenticate": self._challenge_header()},
                    )
                )
                await writer.drain()
                return

            writer.write(
                _http_response(
                    status=200,
                    headers={"Content-Type": f"multipart/mixed; boundary={DEVICE_BOUNDARY}"},
                )
            )
            await writer.drain()

            setup = await read_multipart_part(
                reader,
                boundary=CLIENT_BOUNDARY,
                max_payload_bytes=64 * 1024,
                timeout_s=2.0,
            )
            self.setup_parts.append(setup)
            writer.write(self._setup_response_part())
            await writer.drain()

            while True:
                try:
                    part = await read_multipart_part(
                        reader,
                        boundary=CLIENT_BOUNDARY,
                        max_payload_bytes=1024 * 1024,
                        timeout_s=0.5,
                    )
                except (asyncio.TimeoutError, asyncio.IncompleteReadError):
                    break
                except Exception:
                    break
                if part.header("content-type") == "audio/mp2t":
                    self.audio_parts.append(part)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()
            self.closed_connections += 1
            if task is not None:
                self._tasks.discard(task)

    async def _read_request(self, reader: asyncio.StreamReader) -> FakeTapoHTTPRequest:
        head = await reader.readuntil(_HEADER_SEPARATOR)
        lines = head.decode("iso-8859-1").split("\r\n")
        method, uri, _version = lines[0].split(" ", maxsplit=2)
        headers: dict[str, str] = {}
        for line in lines[1:]:
            name, separator, value = line.partition(":")
            if separator:
                headers[name.strip().lower()] = value.strip()
        request = FakeTapoHTTPRequest(method=method, uri=uri, headers=headers)
        self.requests.append(request)
        return request

    def _authorization_matches(self, request: FakeTapoHTTPRequest) -> bool:
        return digest_authorization_matches(
            authorization_header=request.headers.get("authorization"),
            challenge=self._challenge(),
            method=request.method,
            uri=request.uri,
            username=self.username,
            password_material=self.credential_hash,
        )

    def _challenge_header(self) -> str:
        parts = ['Digest realm="tapo"', 'nonce="fake-nonce"']
        if self.challenge_qop is not None:
            parts.append(f'qop="{self.challenge_qop}"')
        if self.hash_kind == "sha256":
            parts.append('encrypt_type="3"')
        return ", ".join(parts)

    def _challenge(self) -> TapoDigestChallenge:
        return parse_www_authenticate(self._challenge_header())

    def _setup_response_part(self) -> bytes:
        if self.malformed_setup_json:
            body = b"{"
        elif self.omit_session_id:
            body = b'{"params":{}}'
        else:
            body = f'{{"params":{{"session_id":"{self.session_id}"}}}}'.encode()
        return multipart_part(
            DEVICE_PART_PREFIX,
            {"Content-Type": self.setup_content_type},
            body,
        )


def _http_response(
    *,
    status: int,
    reason: str = "OK",
    headers: dict[str, str] | None = None,
    body: bytes = b"",
) -> bytes:
    merged = {"Content-Length": str(len(body))}
    if headers:
        merged.update(headers)
    lines = [f"HTTP/1.1 {status} {reason}"]
    lines.extend(f"{name}: {value}" for name, value in merged.items())
    return ("\r\n".join(lines) + "\r\n\r\n").encode("iso-8859-1") + body
