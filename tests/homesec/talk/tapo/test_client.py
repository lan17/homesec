"""Tests for the Tapo local protocol client."""

from __future__ import annotations

import asyncio
import json
from typing import cast

import pytest

from homesec.models.config import CameraTalkConfig, TalkConfig
from homesec.talk.backends import TalkBackendConfigError, TalkBackendContext
from homesec.talk.tapo import client as tapo_client
from homesec.talk.tapo.client import (
    TapoAuthError,
    TapoLocalClient,
    TapoProtocolError,
    TapoUnsupportedEndpointError,
    _close_writer_best_effort,
    _read_http_response,
    open_tapo_local_client,
)
from homesec.talk.tapo.config import TapoLocalTalkConfig
from homesec.talk.tapo.multipart import (
    CLIENT_PART_PREFIX,
    TapoMultipartError,
    multipart_part,
    read_multipart_part,
)

from .fake_server import FakeTapoServer

_SHA256 = "A" * 64
_MD5 = "B" * 32


def _context(env: dict[str, str]) -> TalkBackendContext:
    return TalkBackendContext(
        camera_name="office",
        source_backend="rtsp",
        runtime_talk=TalkConfig(),
        camera_talk=CameraTalkConfig(backend="tapo_local"),
        resolve_env=lambda name: env.get(name),
    )


@pytest.mark.asyncio
async def test_tapo_client_authenticates_sha256_mode_and_parses_session_id() -> None:
    """Tapo client should authenticate SHA256 Digest mode and parse talk setup."""
    # Given: A fake Tapo endpoint requiring SHA256 password material
    server = FakeTapoServer(hash_kind="sha256", credential_hash=_SHA256)
    await server.start()
    try:
        config = TapoLocalTalkConfig(
            host=server.host,
            port=server.port,
            password_sha256_env="OFFICE_TAPO_SHA256",
        )

        # When: Opening the local Tapo client
        client = await open_tapo_local_client(
            config,
            _context({"OFFICE_TAPO_SHA256": _SHA256.lower()}),
        )
        await client.close()

        # Then: The client authenticated, sent setup JSON, and captured session_id
        assert client.tapo_session_id == server.session_id
        assert server.session_id not in repr(client)
        assert len(server.requests) == 2
        assert server.requests[1].headers["authorization"].startswith("Digest ")
        setup = json.loads(server.setup_parts[0].body.decode("utf-8"))
        assert setup == {
            "params": {"talk": {"mode": "aec"}, "method": "get"},
            "seq": 3,
            "type": "request",
        }
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_tapo_client_rejects_unsupported_digest_qop() -> None:
    """Tapo client should fail closed when Digest qop=auth is unavailable."""
    # Given: A fake endpoint that only advertises an unsupported qop
    server = FakeTapoServer(
        hash_kind="sha256",
        credential_hash=_SHA256,
        challenge_qop="auth-int",
    )
    await server.start()
    try:
        config = TapoLocalTalkConfig(
            host=server.host,
            port=server.port,
            password_sha256_env="OFFICE_TAPO_SHA256",
        )

        # When/Then: Opening fails without sending a legacy no-qop Digest response
        with pytest.raises(TapoProtocolError, match="Digest challenge"):
            await open_tapo_local_client(config, _context({"OFFICE_TAPO_SHA256": _SHA256}))
        assert len(server.requests) == 1
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_fake_tapo_server_rejects_wrong_stream_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fake Tapo endpoint should enforce the local talk stream route."""
    # Given: A fake endpoint and a client accidentally pointed at the wrong path
    server = FakeTapoServer(hash_kind="sha256", credential_hash=_SHA256)
    await server.start()
    monkeypatch.setattr(tapo_client, "_STREAM_URI", "/wrong-stream")
    try:
        config = TapoLocalTalkConfig(
            host=server.host,
            port=server.port,
            password_sha256_env="OFFICE_TAPO_SHA256",
        )

        # When/Then: Opening the client fails before Digest auth succeeds
        with pytest.raises(TapoUnsupportedEndpointError):
            await open_tapo_local_client(config, _context({"OFFICE_TAPO_SHA256": _SHA256}))
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_tapo_client_accepts_json_setup_response_with_charset() -> None:
    """Tapo client should accept JSON setup parts with content-type parameters."""
    # Given: A fake Tapo endpoint that includes a charset on setup JSON
    server = FakeTapoServer(
        hash_kind="sha256",
        credential_hash=_SHA256,
        setup_content_type="application/json; charset=utf-8",
    )
    await server.start()
    try:
        config = TapoLocalTalkConfig(
            host=server.host,
            port=server.port,
            password_sha256_env="OFFICE_TAPO_SHA256",
        )

        # When: Opening the local Tapo client
        client = await open_tapo_local_client(config, _context({"OFFICE_TAPO_SHA256": _SHA256}))
        await client.close()

        # Then: The setup response is parsed as JSON and returns the Tapo session id
        assert client.tapo_session_id == server.session_id
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_tapo_client_authenticates_md5_mode() -> None:
    """Tapo client should authenticate legacy MD5 Digest mode when hinted."""
    # Given: A fake Tapo endpoint without encrypt_type=3
    server = FakeTapoServer(hash_kind="md5", credential_hash=_MD5)
    await server.start()
    try:
        config = TapoLocalTalkConfig(
            host=server.host,
            port=server.port,
            password_md5_env="OFFICE_TAPO_MD5",
        )

        # When: Opening the local Tapo client
        client = await open_tapo_local_client(config, _context({"OFFICE_TAPO_MD5": _MD5}))
        await client.close()

        # Then: The session setup succeeds through MD5 credential material
        assert client.tapo_session_id == server.session_id
        assert len(server.setup_parts) == 1
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_tapo_client_missing_hash_env_maps_to_config_error() -> None:
    """Missing Tapo credential env vars should fail as structured config errors."""
    # Given: A fake Tapo endpoint and config pointing at an unset env var
    server = FakeTapoServer(hash_kind="sha256", credential_hash=_SHA256)
    await server.start()
    try:
        config = TapoLocalTalkConfig(
            host=server.host,
            port=server.port,
            password_sha256_env="TAPO_PASSWORD_SHA256",
        )

        # When: Opening the local Tapo client
        # Then: The error names the missing env var without exposing secret material
        with pytest.raises(TalkBackendConfigError) as exc_info:
            await open_tapo_local_client(config, _context({}))
        assert exc_info.value.public_message == (
            "Required Tapo local environment variable is not set: TAPO_PASSWORD_SHA256"
        )
        assert _SHA256 not in exc_info.value.public_message
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_tapo_client_wrong_hash_maps_to_auth_error_without_hash_value() -> None:
    """Rejected Digest credentials should map to a generic auth error."""
    # Given: A fake Tapo endpoint and a wrong configured hash
    server = FakeTapoServer(hash_kind="sha256", credential_hash=_SHA256)
    await server.start()
    try:
        wrong_hash = "C" * 64
        config = TapoLocalTalkConfig(
            host=server.host,
            port=server.port,
            password_sha256_env="OFFICE_TAPO_SHA256",
        )

        # When: Opening the local Tapo client
        # Then: Authentication fails without exposing the hash value
        with pytest.raises(TapoAuthError) as exc_info:
            await open_tapo_local_client(config, _context({"OFFICE_TAPO_SHA256": wrong_hash}))
        assert str(exc_info.value) == "Tapo local authentication failed"
        assert wrong_hash not in str(exc_info.value)
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_tapo_client_rejects_missing_setup_session_id() -> None:
    """Tapo talk setup should require a non-empty session id."""
    # Given: A fake Tapo endpoint whose setup response omits session_id
    server = FakeTapoServer(
        hash_kind="sha256",
        credential_hash=_SHA256,
        omit_session_id=True,
    )
    await server.start()
    try:
        config = TapoLocalTalkConfig(
            host=server.host,
            port=server.port,
            password_sha256_env="OFFICE_TAPO_SHA256",
        )

        # When: Opening the local Tapo client
        # Then: The malformed setup response is rejected safely
        with pytest.raises(TapoProtocolError, match="session id"):
            await open_tapo_local_client(config, _context({"OFFICE_TAPO_SHA256": _SHA256}))
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_tapo_client_writes_audio_mp2t_multipart_chunk() -> None:
    """Tapo client should write audio/mp2t chunks with the negotiated session id."""
    # Given: An authenticated Tapo client and fake endpoint
    server = FakeTapoServer(hash_kind="sha256", credential_hash=_SHA256)
    await server.start()
    try:
        config = TapoLocalTalkConfig(
            host=server.host,
            port=server.port,
            password_sha256_env="OFFICE_TAPO_SHA256",
        )
        client = await open_tapo_local_client(config, _context({"OFFICE_TAPO_SHA256": _SHA256}))

        # When: Writing one MPEG-TS audio payload
        await client.write_audio_mp2t(b"G" * 188)
        await _wait_for_audio_part(server)
        await client.close()

        # Then: The fake endpoint receives the expected Tapo audio multipart headers
        part = server.audio_parts[0]
        assert part.header("content-type") == "audio/mp2t"
        assert part.header("x-if-encrypt") == "0"
        assert part.header("x-session-id") == server.session_id
        assert part.body == b"G" * 188
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_tapo_client_closes_after_audio_write_failure() -> None:
    """Audio write failures should close the client stream before surfacing."""
    # Given: A connected client whose writer fails during drain
    writer = _FailingDrainWriter()
    client = TapoLocalClient(
        host="127.0.0.1",
        port=8800,
        io_timeout_s=1.0,
        reader=asyncio.StreamReader(),
        writer=cast(asyncio.StreamWriter, writer),
        tapo_session_id="session-token",
    )

    # When: Writing an audio chunk fails
    # Then: The stream is closed and future writes fail as closed-client errors
    with pytest.raises(ConnectionResetError):
        await client.write_audio_mp2t(b"G" * 188)
    assert writer.closed is True
    assert writer.wait_closed_called is True
    with pytest.raises(TapoProtocolError, match="closed"):
        await client.write_audio_mp2t(b"G" * 188)


@pytest.mark.asyncio
async def test_close_writer_best_effort_preserves_cancellation() -> None:
    """Best-effort Tapo close should not swallow task cancellation."""
    # Given: A writer whose close wait is cancelled by the event loop
    writer = _CancelledWaitClosedWriter()

    # When/Then: The helper closes the writer but propagates cancellation
    with pytest.raises(asyncio.CancelledError):
        await _close_writer_best_effort(
            cast(asyncio.StreamWriter, writer),
            timeout_s=1.0,
        )
    assert writer.closed is True


@pytest.mark.asyncio
async def test_multipart_parser_rejects_oversized_payload() -> None:
    """Multipart parser should reject parts over the configured payload limit."""
    # Given: A multipart part whose content length exceeds the caller limit
    reader = asyncio.StreamReader()
    reader.feed_data(
        multipart_part(
            CLIENT_PART_PREFIX,
            {"Content-Type": "application/json"},
            b"{}",
        )
    )

    # When/Then: Reading with a smaller limit fails before accepting the payload
    with pytest.raises(TapoMultipartError, match="exceeds"):
        await read_multipart_part(
            reader,
            boundary="--client-stream-boundary--",
            max_payload_bytes=1,
            timeout_s=1.0,
        )


@pytest.mark.asyncio
async def test_multipart_parser_rejects_oversized_preamble() -> None:
    """Multipart parser should bound preamble/header scanning before payload reads."""
    # Given: A stream that keeps sending non-boundary preamble lines
    reader = asyncio.StreamReader()
    reader.feed_data((b"x" * 1024 + b"\r\n") * 9)

    # When/Then: Reading a part fails once the preamble exceeds the configured limit
    with pytest.raises(TapoMultipartError, match="preamble"):
        await read_multipart_part(
            reader,
            boundary="--client-stream-boundary--",
            max_payload_bytes=64 * 1024,
            timeout_s=1.0,
        )


@pytest.mark.asyncio
async def test_http_response_parser_rejects_malformed_response() -> None:
    """HTTP response parser should reject malformed status lines safely."""
    # Given: A stream containing invalid HTTP response bytes
    reader = asyncio.StreamReader()
    reader.feed_data(b"not-http\r\n\r\n")

    # When/Then: Parsing the response fails with a safe protocol error
    with pytest.raises(TapoProtocolError, match="Malformed"):
        await _read_http_response(reader, io_timeout_s=1.0, read_body=True)


@pytest.mark.asyncio
async def test_http_response_parser_rejects_oversized_body_before_read() -> None:
    """HTTP response parser should cap optional response body reads."""
    # Given: A response that advertises a body larger than the client limit
    reader = asyncio.StreamReader()
    reader.feed_data(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 65537\r\n\r\n")

    # When/Then: Parsing fails without attempting an unbounded read
    with pytest.raises(TapoProtocolError, match="body exceeds"):
        await _read_http_response(reader, io_timeout_s=1.0, read_body=True)


async def _wait_for_audio_part(server: FakeTapoServer) -> None:
    for _ in range(20):
        if server.audio_parts:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("fake Tapo server did not receive an audio part")


class _FailingDrainWriter:
    def __init__(self) -> None:
        self.closed = False
        self.wait_closed_called = False
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        raise ConnectionResetError("simulated drain failure")

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        self.wait_closed_called = True


class _CancelledWaitClosedWriter:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        raise asyncio.CancelledError
