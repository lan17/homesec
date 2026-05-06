from __future__ import annotations

import base64
import hashlib

import pytest

from homesec.sources.rtsp.talk.errors import RTSPProtocolError
from homesec.sources.rtsp.talk.rtsp_auth import (
    RTSPCredentials,
    build_authorization_header,
    parse_www_authenticate,
    redact_rtsp_url,
    request_uri_without_credentials,
    split_rtsp_url_credentials,
)
from homesec.sources.rtsp.talk.rtsp_client import (
    RTSPClient,
    RTSPConnectionConfig,
    RTSPResponse,
    parse_interleaved_channels,
)


def _md5(value: str) -> str:
    return hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()


def test_basic_auth_header() -> None:
    # Given: A Basic authentication challenge and camera credentials
    challenge = parse_www_authenticate('Basic realm="camera"')

    # When: Building the RTSP Authorization header
    header = build_authorization_header(
        challenge=challenge,
        method="DESCRIBE",
        uri="rtsp://camera.local/stream",
        credentials=RTSPCredentials(username="user", password="pass"),
    )

    # Then: The header contains the expected base64 username/password pair
    assert header == "Basic " + base64.b64encode(b"user:pass").decode("ascii")


def test_digest_auth_matches_rfc_2617_vector() -> None:
    # Given: The RFC 2617 Digest authentication example challenge
    challenge = parse_www_authenticate(
        'Digest realm="testrealm@host.com", '
        'qop="auth,auth-int", '
        'nonce="dcd98b7102dd2f0e8b11d0f600bfb0c093", '
        'opaque="5ccc069c403ebaf9f0171e9517f40e41"'
    )

    # When: Building a Digest Authorization header for the RFC request vector
    header = build_authorization_header(
        challenge=challenge,
        method="GET",
        uri="/dir/index.html",
        credentials=RTSPCredentials(username="Mufasa", password="Circle Of Life"),
        nonce_count=1,
        cnonce="0a4f113b",
    )

    # Then: The generated digest fields match the RFC response vector
    assert 'response="6629fae49393a05397450978507c4ef1"' in header
    assert 'opaque="5ccc069c403ebaf9f0171e9517f40e41"' in header
    assert "qop=auth" in header
    assert "nc=00000001" in header


def test_digest_auth_supports_md5_sess() -> None:
    # Given: A Digest challenge using the MD5-sess algorithm
    challenge = parse_www_authenticate(
        'Digest realm="camera", nonce="abc", algorithm=MD5-sess, qop="auth"'
    )
    method = "ANNOUNCE"
    uri = "rtsp://camera.local/talk"
    credentials = RTSPCredentials(username="alice", password="secret")
    ha1 = _md5(f"{credentials.username}:camera:{credentials.password}")
    ha1_sess = _md5(f"{ha1}:abc:clientnonce")
    ha2 = _md5(f"{method}:{uri}")
    expected = _md5(f"{ha1_sess}:abc:00000002:clientnonce:auth:{ha2}")

    # When: Building the Authorization header with a fixed nonce count and cnonce
    header = build_authorization_header(
        challenge=challenge,
        method=method,
        uri=uri,
        credentials=credentials,
        nonce_count=2,
        cnonce="clientnonce",
    )

    # Then: The response digest uses the MD5-sess calculation
    assert "algorithm=MD5-sess" in header
    assert f'response="{expected}"' in header
    assert "nc=00000002" in header


def test_digest_auth_supports_no_qop_challenge() -> None:
    # Given: A legacy Digest challenge without qop
    challenge = parse_www_authenticate('Digest realm="camera", nonce="abc"')
    method = "DESCRIBE"
    uri = "rtsp://camera.local/live"
    credentials = RTSPCredentials(username="alice", password="secret")
    expected = _md5(
        f"{_md5(f'{credentials.username}:camera:{credentials.password}')}:abc:{_md5(f'{method}:{uri}')}"
    )

    # When: Building the Authorization header
    header = build_authorization_header(
        challenge=challenge,
        method=method,
        uri=uri,
        credentials=credentials,
    )

    # Then: The legacy digest omits qop-specific fields
    assert f'response="{expected}"' in header
    assert "qop=" not in header
    assert "nc=" not in header
    assert "cnonce=" not in header


@pytest.mark.parametrize(
    ("header", "match"),
    [
        ('Bearer realm="camera"', "Unsupported RTSP auth scheme"),
        ('Digest nonce="abc"', "realm and nonce"),
        ('Digest realm="camera"', "realm and nonce"),
        ('Digest realm="camera", nonce="abc", algorithm=SHA-256', "algorithm"),
    ],
)
def test_authorization_rejects_unsupported_or_incomplete_challenges(
    header: str,
    match: str,
) -> None:
    # Given: An unsupported or incomplete RTSP authentication challenge
    challenge = parse_www_authenticate(header)

    # When: Building an Authorization header from that challenge
    # Then: Validation rejects it before sending credentials
    with pytest.raises(ValueError, match=match):
        build_authorization_header(
            challenge=challenge,
            method="DESCRIBE",
            uri="rtsp://camera.local/live",
            credentials=RTSPCredentials(username="alice", password="secret"),
        )


def test_parse_authenticate_without_parameters() -> None:
    # Given: An Authenticate header that has only a scheme
    challenge = parse_www_authenticate("Negotiate")

    # When: Parsing the header value
    # Then: The scheme is kept and optional parameters remain empty
    assert challenge.scheme == "negotiate"
    assert challenge.params == {}
    assert challenge.realm is None


def test_rtsp_url_credentials_are_split_and_redacted() -> None:
    # Given: An RTSP URL containing percent-encoded credentials
    clean, credentials = split_rtsp_url_credentials("rtsp://alice:s3cr%40t@camera.local:8554/live")

    # When: Splitting and redacting URL credentials
    # Then: Helpers produce credential-free camera URLs and decoded credentials
    assert clean == "rtsp://camera.local:8554/live"
    assert credentials == RTSPCredentials(username="alice", password="s3cr@t")
    assert redact_rtsp_url("rtsp://alice:s3cr%40t@camera.local:8554/live") == (
        "rtsp://camera.local:8554/live"
    )
    assert request_uri_without_credentials("rtsp://alice:pw@camera.local/live") == (
        "rtsp://camera.local/live"
    )


def test_rtsp_url_helpers_preserve_ipv6_hosts() -> None:
    # Given: An RTSP URL with credentials and an IPv6 host literal
    url = "rtsp://alice:p%40ss@[2001:db8::1]:8554/live?profile=main"

    # When: Splitting and redacting URL credentials
    clean, credentials = split_rtsp_url_credentials(url)

    # Then: The IPv6 host and query string are preserved without credentials
    assert clean == "rtsp://[2001:db8::1]:8554/live?profile=main"
    assert credentials == RTSPCredentials(username="alice", password="p@ss")
    assert redact_rtsp_url(url) == "rtsp://[2001:db8::1]:8554/live?profile=main"


def test_rtsp_url_helpers_allow_urls_without_credentials() -> None:
    # Given: An RTSP URL that contains no credentials
    url = "rtsp://camera.local/live"

    # When: Splitting and redacting URL credentials
    # Then: Credential helpers leave the URL unchanged and report no credentials
    assert split_rtsp_url_credentials(url) == (url, None)
    assert redact_rtsp_url(url) == url


def test_rtsp_response_parser_handles_headers_and_body() -> None:
    # Given: A raw RTSP response with headers, declared body length, and extra bytes
    raw = (
        b"RTSP/1.0 200 OK\r\n"
        b"CSeq: 3\r\n"
        b"Session: abc;timeout=60\r\n"
        b"Content-Length: 4\r\n"
        b"\r\n"
        b"testextra"
    )

    # When: Parsing the response bytes
    response = RTSPResponse.parse(raw)

    # Then: Headers are normalized and the body honors Content-Length
    assert response.status_code == 200
    assert response.reason == "OK"
    assert response.header("session") == "abc;timeout=60"
    assert response.body == b"test"


def test_rtsp_response_parser_combines_duplicate_headers() -> None:
    # Given: A raw RTSP response containing duplicate headers
    raw = b"RTSP/1.0 200 OK\r\nX-Test: one\r\nX-Test: two\r\n\r\n"

    # When: Parsing the response bytes
    response = RTSPResponse.parse(raw)

    # Then: Duplicate values are combined under the same header key
    assert response.header("X-Test") == "one, two"


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (b"RTSP/1.0 200 OK\r\n", "header terminator"),
        (b"HTTP/1.1 200 OK\r\n\r\n", "status line"),
        (b"RTSP/1.0 200 OK\r\nnot-a-header\r\n\r\n", "header line"),
        (b"RTSP/1.0 200 OK\r\nContent-Length: nope\r\n\r\n", "Content-Length"),
        (b"RTSP/1.0 200 OK\r\nContent-Length: -1\r\n\r\nbody", "Content-Length"),
        (b"RTSP/1.0 200 OK\r\nContent-Length: 4\r\n\r\na", "shorter"),
    ],
)
def test_rtsp_response_parser_rejects_malformed_responses(raw: bytes, match: str) -> None:
    # Given: Malformed raw RTSP response bytes
    # When: Parsing the response
    # Then: The parser rejects the malformed shape with a protocol error
    with pytest.raises(RTSPProtocolError, match=match):
        RTSPResponse.parse(raw)


def test_parse_interleaved_channels() -> None:
    # Given: RTSP Transport header values with and without interleaved channel ranges
    # When: Parsing interleaved channels from each transport value
    # Then: Valid ranges are extracted and UDP transport is ignored
    assert parse_interleaved_channels("RTP/AVP/TCP;unicast;interleaved=2-3") == (2, 3)
    assert parse_interleaved_channels("RTP/AVP/TCP; interleaved = 10 - 11") == (10, 11)
    assert parse_interleaved_channels("RTP/AVP;unicast;client_port=10-11") is None


@pytest.mark.asyncio
async def test_rtsp_client_connect_rejects_invalid_urls() -> None:
    # Given: RTSP clients configured with non-RTSP or hostless URLs
    # When: Connecting each invalid client
    # Then: URL validation rejects the invalid URL before opening a socket
    with pytest.raises(ValueError, match="rtsp://"):
        await RTSPClient(RTSPConnectionConfig(url="http://camera.local/live")).connect()

    with pytest.raises(ValueError, match="host"):
        await RTSPClient(RTSPConnectionConfig(url="rtsp:///live")).connect()


@pytest.mark.asyncio
async def test_rtsp_client_requires_connection_before_requests_or_frames() -> None:
    # Given: An RTSP client that has not connected to a camera
    client = RTSPClient(RTSPConnectionConfig(url="rtsp://camera.local/live"))

    # When: Requests and interleaved writes are attempted before connection
    # Then: The client rejects them without touching a socket
    with pytest.raises(RuntimeError, match="not connected"):
        await client.describe()

    with pytest.raises(RuntimeError, match="not connected"):
        await client.send_interleaved_frame(0, b"payload")

    # Then: Closing an unconnected client remains safe
    await client.close()


@pytest.mark.asyncio
async def test_rtsp_client_validates_interleaved_frame_shape_before_write() -> None:
    # Given: An RTSP client with frame validation independent of socket state
    client = RTSPClient(RTSPConnectionConfig(url="rtsp://camera.local/live"))

    # When: Sending invalid interleaved frame values
    # Then: Invalid channel and oversized payload values are rejected
    with pytest.raises(ValueError, match="range"):
        await client.send_interleaved_frame(-1, b"payload")

    with pytest.raises(ValueError, match="65535"):
        await client.send_interleaved_frame(0, b"x" * 0x10000)
