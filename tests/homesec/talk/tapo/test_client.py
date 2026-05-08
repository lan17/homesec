"""Tests for the Tapo local protocol client."""

from __future__ import annotations

import asyncio
import json

import pytest

from homesec.models.config import CameraTalkConfig, TalkConfig
from homesec.talk.backends import TalkBackendConfigError, TalkBackendContext
from homesec.talk.tapo.client import (
    TapoAuthError,
    TapoProtocolError,
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
async def test_http_response_parser_rejects_malformed_response() -> None:
    """HTTP response parser should reject malformed status lines safely."""
    # Given: A stream containing invalid HTTP response bytes
    reader = asyncio.StreamReader()
    reader.feed_data(b"not-http\r\n\r\n")

    # When/Then: Parsing the response fails with a safe protocol error
    with pytest.raises(TapoProtocolError, match="Malformed"):
        await _read_http_response(reader, io_timeout_s=1.0, read_body=True)


async def _wait_for_audio_part(server: FakeTapoServer) -> None:
    for _ in range(20):
        if server.audio_parts:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("fake Tapo server did not receive an audio part")
