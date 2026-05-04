from __future__ import annotations

import base64

from homesec.sources.rtsp.talk.rtsp_auth import (
    RTSPCredentials,
    build_authorization_header,
    parse_www_authenticate,
    redact_rtsp_url,
    split_rtsp_url_credentials,
)
from homesec.sources.rtsp.talk.rtsp_client import RTSPResponse, parse_interleaved_channels


def test_basic_auth_header() -> None:
    challenge = parse_www_authenticate('Basic realm="camera"')

    header = build_authorization_header(
        challenge=challenge,
        method="DESCRIBE",
        uri="rtsp://camera.local/stream",
        credentials=RTSPCredentials(username="user", password="pass"),
    )

    assert header == "Basic " + base64.b64encode(b"user:pass").decode("ascii")


def test_digest_auth_matches_rfc_2617_vector() -> None:
    challenge = parse_www_authenticate(
        'Digest realm="testrealm@host.com", '
        'qop="auth,auth-int", '
        'nonce="dcd98b7102dd2f0e8b11d0f600bfb0c093", '
        'opaque="5ccc069c403ebaf9f0171e9517f40e41"'
    )

    header = build_authorization_header(
        challenge=challenge,
        method="GET",
        uri="/dir/index.html",
        credentials=RTSPCredentials(username="Mufasa", password="Circle Of Life"),
        nonce_count=1,
        cnonce="0a4f113b",
    )

    assert 'response="6629fae49393a05397450978507c4ef1"' in header
    assert "qop=auth" in header
    assert "nc=00000001" in header


def test_rtsp_url_credentials_are_split_and_redacted() -> None:
    clean, credentials = split_rtsp_url_credentials("rtsp://alice:s3cr%40t@camera.local:8554/live")

    assert clean == "rtsp://camera.local:8554/live"
    assert credentials == RTSPCredentials(username="alice", password="s3cr@t")
    assert redact_rtsp_url("rtsp://alice:s3cr%40t@camera.local:8554/live") == (
        "rtsp://camera.local:8554/live"
    )


def test_rtsp_response_parser_handles_headers_and_body() -> None:
    raw = (
        b"RTSP/1.0 200 OK\r\n"
        b"CSeq: 3\r\n"
        b"Session: abc;timeout=60\r\n"
        b"Content-Length: 4\r\n"
        b"\r\n"
        b"testextra"
    )

    response = RTSPResponse.parse(raw)

    assert response.status_code == 200
    assert response.reason == "OK"
    assert response.header("session") == "abc;timeout=60"
    assert response.body == b"test"


def test_parse_interleaved_channels() -> None:
    assert parse_interleaved_channels("RTP/AVP/TCP;unicast;interleaved=2-3") == (2, 3)
    assert parse_interleaved_channels("RTP/AVP;unicast;client_port=10-11") is None
