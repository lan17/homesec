from __future__ import annotations

import asyncio
import base64
import hashlib
import re
import struct
from dataclasses import dataclass, field

import pytest

from homesec.models.talk import TalkCapabilityState, TalkRefusalReason
from homesec.sources.rtsp.talk.errors import (
    CameraBackchannelUnsupportedError,
    CameraRejectedTalkSessionError,
    CameraTalkStreamFailedError,
    RTSPAuthenticationError,
    RTSPProtocolError,
    UnsupportedTalkCodecError,
)
from homesec.sources.rtsp.talk.g711 import encode_pcma
from homesec.sources.rtsp.talk.models import ONVIFBackchannelConfig
from homesec.sources.rtsp.talk.onvif_backchannel import ONVIFBackchannelAdapter, ONVIFTalkSession
from homesec.sources.rtsp.talk.resample import resample_pcm_s16le_mono
from homesec.sources.rtsp.talk.rtp import parse_rtp_header
from homesec.sources.rtsp.talk.rtsp_client import RTSPResponse
from homesec.sources.rtsp.talk.sdp import SDPCodec, SDPMediaDescription, SelectedBackchannel

_BACKCHANNEL_SDP = """v=0\r
o=- 0 0 IN IP4 127.0.0.1\r
s=HomeSec fake backchannel camera\r
t=0 0\r
m=video 0 RTP/AVP 96\r
a=recvonly\r
a=rtpmap:96 H264/90000\r
a=control:trackID=video\r
m=audio 0 RTP/AVP 0\r
a=sendonly\r
a=rtpmap:0 PCMU/8000\r
a=control:trackID=backchannel\r
"""


@dataclass(slots=True)
class _RTSPRequest:
    method: str
    uri: str
    headers: dict[str, str]


@dataclass(slots=True)
class _FakeRTSPBackchannelServer:
    describe_status: int = 200
    setup_status: int = 200
    play_status: int = 200
    teardown_status: int = 200
    require_basic_auth: bool = False
    require_digest_auth: bool = False
    digest_qop: str | None = "auth"
    basic_username: str = "alice"
    basic_password: str = "s3cr@t"
    transport_header: str = "RTP/AVP/TCP;unicast;interleaved=0-1"
    setup_session_header: str | None = "homesec-talk;timeout=60"
    content_base_header: str | None = None
    content_location_header: str | None = None
    interleaved_before_response_methods: set[str] = field(default_factory=set)
    sdp_body: bytes = field(default_factory=lambda: _BACKCHANNEL_SDP.encode())
    requests: list[_RTSPRequest] = field(default_factory=list)
    interleaved_before_play: list[tuple[int, bytes]] = field(default_factory=list)
    interleaved_after_play: list[tuple[int, bytes]] = field(default_factory=list)
    _play_seen: bool = False
    _server: asyncio.AbstractServer | None = None
    _port: int | None = None

    @property
    def url(self) -> str:
        assert self._port is not None
        return f"rtsp://127.0.0.1:{self._port}/Streaming/Channels/101"

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, "127.0.0.1", 0)
        socket = self._server.sockets[0]
        host, port = socket.getsockname()[:2]
        assert host == "127.0.0.1"
        self._port = int(port)

    async def stop(self) -> None:
        server = self._server
        if server is None:
            return
        server.close()
        await server.wait_closed()
        self._server = None

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while True:
                first = await reader.readexactly(1)
                if first == b"$":
                    channel = (await reader.readexactly(1))[0]
                    length = int.from_bytes(await reader.readexactly(2), "big")
                    payload = await reader.readexactly(length)
                    target = (
                        self.interleaved_after_play
                        if self._play_seen
                        else self.interleaved_before_play
                    )
                    target.append((channel, payload))
                    continue

                request = await self._read_request(first, reader)
                self.requests.append(request)
                response = self._response_for(request)
                if request.method in self.interleaved_before_response_methods:
                    writer.write(_interleaved_frame(channel=1, payload=b"rtcp"))
                writer.write(response)
                await writer.drain()
                if request.method == "PLAY" and self.play_status == 200:
                    self._play_seen = True
                if request.method == "TEARDOWN":
                    break
        except asyncio.IncompleteReadError:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def _read_request(
        self,
        first: bytes,
        reader: asyncio.StreamReader,
    ) -> _RTSPRequest:
        head = first + await reader.readuntil(b"\r\n\r\n")
        text = head.decode("iso-8859-1")
        lines = text.split("\r\n")
        method, uri, _version = lines[0].split(" ", maxsplit=2)
        headers: dict[str, str] = {}
        for line in lines[1:]:
            name, separator, value = line.partition(":")
            if separator:
                headers[name.lower()] = value.strip()
        content_length = int(headers.get("content-length", "0") or "0")
        if content_length:
            await reader.readexactly(content_length)
        return _RTSPRequest(method=method, uri=uri, headers=headers)

    def _digest_challenge_header(self) -> str:
        parts = ['Digest realm="homesec-fake"', 'nonce="digest-nonce"']
        if self.digest_qop is not None:
            parts.append(f'qop="{self.digest_qop}"')
        return ", ".join(parts)

    def _response_for(self, request: _RTSPRequest) -> bytes:
        cseq = request.headers.get("cseq", "1")
        if request.method == "DESCRIBE":
            if self.require_basic_auth:
                expected = base64.b64encode(
                    f"{self.basic_username}:{self.basic_password}".encode()
                ).decode("ascii")
                if request.headers.get("authorization") != f"Basic {expected}":
                    return _rtsp_response(
                        cseq=cseq,
                        status=401,
                        reason="Unauthorized",
                        headers={"WWW-Authenticate": 'Basic realm="homesec-fake"'},
                    )
            if self.require_digest_auth and not _digest_authorization_matches(request, self):
                return _rtsp_response(
                    cseq=cseq,
                    status=401,
                    reason="Unauthorized",
                    headers={"WWW-Authenticate": self._digest_challenge_header()},
                )
            if self.describe_status != 200:
                return _rtsp_response(
                    cseq=cseq, status=self.describe_status, reason="Option Not Supported"
                )
            headers = {"Content-Type": "application/sdp"}
            if self.content_base_header is not None:
                headers["Content-Base"] = self.content_base_header
            if self.content_location_header is not None:
                headers["Content-Location"] = self.content_location_header
            return _rtsp_response(
                cseq=cseq,
                headers=headers,
                body=self.sdp_body,
            )
        if request.method == "SETUP":
            if self.setup_status != 200:
                return _rtsp_response(cseq=cseq, status=self.setup_status, reason="Setup Failed")
            headers = {"Transport": self.transport_header}
            if self.setup_session_header is not None:
                headers["Session"] = self.setup_session_header
            return _rtsp_response(
                cseq=cseq,
                headers=headers,
            )
        if request.method == "PLAY":
            if self.play_status == 200:
                return _rtsp_response(cseq=cseq, headers={"Session": "homesec-talk"})
            return _rtsp_response(cseq=cseq, status=self.play_status, reason="Play Failed")
        if request.method == "TEARDOWN":
            return _rtsp_response(
                cseq=cseq,
                status=self.teardown_status,
                reason="Teardown Failed" if self.teardown_status != 200 else "OK",
                headers={"Session": "homesec-talk"},
            )
        return _rtsp_response(cseq=cseq, status=405, reason="Method Not Allowed")


class _NoopTalkClient:
    def __init__(self, *, session_id: str | None = None) -> None:
        self.session_id = session_id
        self.sent: list[tuple[int, bytes]] = []
        self.teardown_calls = 0
        self.close_calls = 0

    async def send_interleaved_frame(self, channel: int, payload: bytes) -> None:
        self.sent.append((channel, payload))

    async def teardown(self, *, headers: dict[str, str] | None = None) -> None:
        self.teardown_calls += 1

    async def close(self) -> None:
        self.close_calls += 1


class _FailingSendTalkClient(_NoopTalkClient):
    async def send_interleaved_frame(self, channel: int, payload: bytes) -> None:
        raise OSError("broken pipe")


class _FailingTeardownTalkClient(_NoopTalkClient):
    async def teardown(self, *, headers: dict[str, str] | None = None) -> None:
        self.teardown_calls += 1
        raise RuntimeError("teardown failed")


def _rtsp_response(
    *,
    cseq: str,
    status: int = 200,
    reason: str = "OK",
    headers: dict[str, str] | None = None,
    body: bytes = b"",
) -> bytes:
    merged = {"CSeq": cseq, "Content-Length": str(len(body))}
    if headers:
        merged.update(headers)
    lines = [f"RTSP/1.0 {status} {reason}"]
    lines.extend(f"{name}: {value}" for name, value in merged.items())
    return ("\r\n".join(lines) + "\r\n\r\n").encode("iso-8859-1") + body


def _interleaved_frame(*, channel: int, payload: bytes) -> bytes:
    return b"$" + bytes([channel]) + len(payload).to_bytes(2, "big") + payload


def _digest_authorization_matches(
    request: _RTSPRequest,
    server: _FakeRTSPBackchannelServer,
) -> bool:
    authorization = request.headers.get("authorization")
    if authorization is None or not authorization.startswith("Digest "):
        return False
    values = _parse_digest_values(authorization.removeprefix("Digest "))
    if values.get("username") != server.basic_username:
        return False
    if values.get("realm") != "homesec-fake" or values.get("nonce") != "digest-nonce":
        return False
    if values.get("uri") != request.uri:
        return False

    algorithm = values.get("algorithm", "MD5")
    if algorithm.upper() != "MD5":
        return False
    ha1 = _md5(f"{server.basic_username}:homesec-fake:{server.basic_password}")
    ha2 = _md5(f"{request.method}:{request.uri}")
    qop = values.get("qop")
    if qop:
        expected = _md5(f"{ha1}:digest-nonce:{values.get('nc')}:{values.get('cnonce')}:{qop}:{ha2}")
    else:
        expected = _md5(f"{ha1}:digest-nonce:{ha2}")
    return values.get("response") == expected


def _parse_digest_values(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in re.finditer(r'(\w+)=("(?:[^"\\]|\\.)*"|[^,]+)', value):
        raw = match.group(2).strip()
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        result[match.group(1).lower()] = raw
    return result


def _md5(value: str) -> str:
    return hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()


def _pcm_frame_16khz_20ms() -> bytes:
    samples = [0, 1200] * 160
    return struct.pack(f"<{len(samples)}h", *samples)


def _selected(codec_name: str = "PCMU", payload_type: int = 0) -> SelectedBackchannel:
    media = SDPMediaDescription(
        media="audio",
        port=0,
        proto="RTP/AVP",
        payload_types=[payload_type],
    )
    codec = SDPCodec(
        payload_type=payload_type,
        encoding_name=codec_name,
        clock_rate=8000,
    )
    return SelectedBackchannel(
        control="rtsp://camera.local/talk",
        payload_type=payload_type,
        codec=codec,
        media=media,
    )


@pytest.mark.asyncio
async def test_onvif_backchannel_probe_reports_supported_camera_without_setup() -> None:
    """Probe should classify support from DESCRIBE without opening a talk stream."""
    # Given: A camera that advertises a supported ONVIF audio backchannel
    server = _FakeRTSPBackchannelServer()
    await server.start()
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=server.url),
            camera_name="front_door",
        )

        # When: Probing camera talk capability
        result = await adapter.probe()
    finally:
        await server.stop()

    # Then: Capability is supported and no SETUP/PLAY/RTP happens
    assert result.capability == TalkCapabilityState.SUPPORTED
    assert result.offered_codecs == ["PCMU/8000"]
    assert result.selected_codec == "PCMU/8000"
    assert [request.method for request in server.requests] == ["DESCRIBE"]
    assert server.interleaved_before_play == []
    assert server.interleaved_after_play == []


@pytest.mark.asyncio
async def test_onvif_backchannel_probe_ignores_close_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe classification should not be overwritten by best-effort close failures."""

    class _CloseFailingClient:
        async def connect(self) -> None:
            return None

        async def describe(self, *, headers: dict[str, str] | None = None) -> RTSPResponse:
            _ = headers
            return RTSPResponse(
                status_code=200,
                reason="OK",
                body=_BACKCHANNEL_SDP.encode(),
            )

        async def close(self) -> None:
            raise RuntimeError("camera reset during close")

    # Given: A probe client that returns supported SDP but fails during cleanup
    adapter = ONVIFBackchannelAdapter(
        ONVIFBackchannelConfig(rtsp_url="rtsp://camera.local/stream1"),
        camera_name="front_door",
    )
    monkeypatch.setattr(adapter, "_client", lambda: _CloseFailingClient())

    # When: Probing camera talk capability
    result = await adapter.probe()

    # Then: The supported result is preserved despite cleanup failure
    assert result.capability == TalkCapabilityState.SUPPORTED
    assert result.selected_codec == "PCMU/8000"


@pytest.mark.asyncio
async def test_onvif_backchannel_probe_reports_unsupported_codec() -> None:
    """Probe should expose camera codecs when HomeSec cannot encode them."""
    # Given: A camera that advertises only an unsupported backchannel codec
    server = _FakeRTSPBackchannelServer(
        sdp_body=_BACKCHANNEL_SDP.replace("PCMU/8000", "OPUS/48000")
        .replace("RTP/AVP 0", "RTP/AVP 96")
        .replace("rtpmap:0", "rtpmap:96")
        .encode()
    )
    await server.start()
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=server.url),
            camera_name="front_door",
        )

        # When: Probing camera talk capability
        result = await adapter.probe()
    finally:
        await server.stop()

    # Then: The result is an unsupported-codec capability, not a protocol crash
    assert result.capability == TalkCapabilityState.UNSUPPORTED_CODEC
    assert result.refusal_reason == TalkRefusalReason.UNSUPPORTED_CODEC
    assert result.offered_codecs == ["OPUS/48000"]
    assert result.message is not None
    assert "advertised: OPUS/48000" in result.message


@pytest.mark.asyncio
async def test_onvif_backchannel_probe_reports_missing_backchannel() -> None:
    """Probe should distinguish cameras without sendonly audio backchannel support."""
    # Given: A camera that rejects the ONVIF backchannel DESCRIBE extension
    server = _FakeRTSPBackchannelServer(describe_status=551)
    await server.start()
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=server.url),
            camera_name="front_door",
        )

        # When: Probing camera talk capability
        result = await adapter.probe()
    finally:
        await server.stop()

    # Then: The result is an unsupported-camera capability
    assert result.capability == TalkCapabilityState.UNSUPPORTED
    assert result.refusal_reason == TalkRefusalReason.UNSUPPORTED_CAMERA


@pytest.mark.asyncio
async def test_onvif_backchannel_probe_reports_auth_failure() -> None:
    """Probe should distinguish camera auth rejection from retryable protocol failure."""
    # Given: A camera that requires RTSP auth but no talk credentials are configured
    server = _FakeRTSPBackchannelServer(require_basic_auth=True)
    await server.start()
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=server.url),
            camera_name="front_door",
        )

        # When: Probing camera talk capability
        result = await adapter.probe()
    finally:
        await server.stop()

    # Then: Capability is an auth failure and no SETUP/PLAY/RTP happens
    assert result.capability == TalkCapabilityState.ERROR
    assert result.refusal_reason == TalkRefusalReason.TALK_AUTH_FAILED
    assert result.message == "Camera talk backchannel authentication failed"
    assert [request.method for request in server.requests] == ["DESCRIBE"]
    assert server.interleaved_before_play == []
    assert server.interleaved_after_play == []


@pytest.mark.asyncio
async def test_onvif_backchannel_probe_redacts_protocol_parse_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe errors should not expose raw malformed RTSP protocol payloads."""

    class _ProtocolFailingClient:
        async def connect(self) -> None:
            return None

        async def describe(self, *, headers: dict[str, str] | None = None) -> RTSPResponse:
            _ = headers
            raise RTSPProtocolError(
                "Invalid RTSP status line: 'RTSP/1.0 500 rtsp://alice:secret@camera.local'"
            )

        async def close(self) -> None:
            return None

    adapter = ONVIFBackchannelAdapter(
        ONVIFBackchannelConfig(rtsp_url="rtsp://camera.local/stream1"),
        camera_name="front_door",
    )
    monkeypatch.setattr(adapter, "_client", lambda: _ProtocolFailingClient())

    # Given: A camera returns malformed RTSP data containing sensitive-looking text
    # When: Probing camera talk capability
    result = await adapter.probe()

    # Then: The public capability error is stable and omits the raw protocol payload
    assert result.capability == TalkCapabilityState.ERROR
    assert result.refusal_reason == TalkRefusalReason.CAMERA_BACKCHANNEL_FAILED
    assert result.message == "Camera talk backchannel protocol failed"
    assert "alice:secret" not in (result.message or "")


@pytest.mark.asyncio
async def test_onvif_backchannel_sends_no_rtp_before_play_200_ok() -> None:
    # Given: A fake camera accepting the ONVIF RTSP backchannel handshake
    server = _FakeRTSPBackchannelServer()
    await server.start()
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=server.url),
            camera_name="front_door",
        )
        session = await adapter.open_session(session_id="talk-1")

        # When: Sending one PCM frame after the adapter has completed PLAY
        await session.write_pcm_frame(_pcm_frame_16khz_20ms())
        await session.close()
    finally:
        await server.stop()

    # Then: RTP is emitted only after DESCRIBE, SETUP, and PLAY complete
    assert [request.method for request in server.requests] == [
        "DESCRIBE",
        "SETUP",
        "PLAY",
        "TEARDOWN",
    ]
    assert all(
        request.headers.get("require") == "www.onvif.org/ver20/backchannel"
        for request in server.requests[:3]
    )
    assert server.interleaved_before_play == []
    assert len(server.interleaved_after_play) == 1

    channel, packet = server.interleaved_after_play[0]
    assert channel == 0
    header = parse_rtp_header(packet)
    assert header["version"] == 2
    assert header["payload_type"] == 0
    assert packet[12:]


@pytest.mark.asyncio
async def test_onvif_backchannel_uses_camera_selected_interleaved_channel() -> None:
    # Given: A camera that selects a non-default RTP interleaved channel during SETUP
    server = _FakeRTSPBackchannelServer(
        transport_header="RTP/AVP/TCP;unicast;interleaved=2-3",
    )
    await server.start()
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=server.url),
            camera_name="front_door",
        )
        session = await adapter.open_session(session_id="talk-1")

        # When: Sending a PCM frame through the negotiated session
        await session.write_pcm_frame(_pcm_frame_16khz_20ms())
        await session.close()
    finally:
        await server.stop()

    # Then: HomeSec offers the default channels but writes RTP on the camera-selected channel
    setup_request = next(request for request in server.requests if request.method == "SETUP")
    assert setup_request.headers["transport"] == "RTP/AVP/TCP;unicast;interleaved=0-1"
    assert server.interleaved_after_play[0][0] == 2


@pytest.mark.asyncio
async def test_onvif_backchannel_skips_interleaved_frames_before_control_response() -> None:
    # Given: A camera that sends interleaved bytes before the PLAY control response
    server = _FakeRTSPBackchannelServer(interleaved_before_response_methods={"PLAY"})
    await server.start()
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=server.url),
            camera_name="front_door",
        )

        # When: Opening and closing a talk session without writing audio
        session = await adapter.open_session(session_id="talk-1")
        await session.close()
    finally:
        await server.stop()

    # Then: Pre-response interleaved bytes are ignored and no RTP is recorded as post-PLAY audio
    assert [request.method for request in server.requests] == [
        "DESCRIBE",
        "SETUP",
        "PLAY",
        "TEARDOWN",
    ]
    assert server.interleaved_before_play == []
    assert server.interleaved_after_play == []


@pytest.mark.parametrize("header_attr", ["content_base_header", "content_location_header"])
@pytest.mark.asyncio
async def test_onvif_backchannel_uses_response_base_for_relative_controls(
    header_attr: str,
) -> None:
    # Given: A camera SDP with relative media controls and a response base header
    server = _FakeRTSPBackchannelServer()
    await server.start()
    setattr(server, header_attr, f"{server.url}/response-base/")
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=server.url),
            camera_name="front_door",
        )

        # When: Opening a talk session from that SDP
        session = await adapter.open_session(session_id="talk-1")
        await session.close()
    finally:
        await server.stop()

    # Then: The SETUP URI resolves the relative control against the response base
    setup_request = next(request for request in server.requests if request.method == "SETUP")
    assert setup_request.uri == f"{server.url}/response-base/trackID=backchannel"


@pytest.mark.asyncio
async def test_onvif_backchannel_maps_describe_551_to_unsupported() -> None:
    # Given: A camera that rejects the ONVIF backchannel DESCRIBE requirement
    server = _FakeRTSPBackchannelServer(describe_status=551)
    await server.start()
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=server.url),
            camera_name="front_door",
        )

        # When: Opening a talk session
        with pytest.raises(CameraBackchannelUnsupportedError):
            await adapter.open_session(session_id="talk-1")
    finally:
        await server.stop()

    # Then: The adapter stops at DESCRIBE and reports unsupported backchannel
    assert [request.method for request in server.requests] == ["DESCRIBE"]
    assert server.interleaved_before_play == []
    assert server.interleaved_after_play == []


@pytest.mark.asyncio
async def test_onvif_backchannel_maps_describe_non_551_failure_to_rejected() -> None:
    # Given: A camera that rejects DESCRIBE with a generic RTSP failure
    server = _FakeRTSPBackchannelServer(describe_status=403)
    await server.start()
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=server.url),
            camera_name="front_door",
        )

        # When: Opening a talk session
        with pytest.raises(CameraRejectedTalkSessionError):
            await adapter.open_session(session_id="talk-1")
    finally:
        await server.stop()

    # Then: The adapter classifies the failure as a rejected talk session
    assert [request.method for request in server.requests] == ["DESCRIBE"]
    assert server.interleaved_before_play == []
    assert server.interleaved_after_play == []


@pytest.mark.asyncio
async def test_onvif_backchannel_retries_basic_auth_from_url_credentials() -> None:
    # Given: A camera that requires Basic auth and credentials embedded in the RTSP URL
    server = _FakeRTSPBackchannelServer(require_basic_auth=True)
    await server.start()
    try:
        credentialed_url = server.url.replace("rtsp://", "rtsp://alice:s3cr%40t@", 1)
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=credentialed_url),
            camera_name="front_door",
        )

        # When: Opening and closing a talk session
        session = await adapter.open_session(session_id="talk-1")
        await session.close()
    finally:
        await server.stop()

    # Then: The adapter retries DESCRIBE with redacted URI and Basic credentials
    assert [request.method for request in server.requests] == [
        "DESCRIBE",
        "DESCRIBE",
        "SETUP",
        "PLAY",
        "TEARDOWN",
    ]
    authorization = server.requests[1].headers["authorization"]
    expected = base64.b64encode(b"alice:s3cr@t").decode("ascii")
    assert authorization == f"Basic {expected}"
    assert "@" not in server.requests[1].uri
    assert server.interleaved_before_play == []
    assert server.interleaved_after_play == []


@pytest.mark.parametrize("digest_qop", [None, "auth"])
@pytest.mark.asyncio
async def test_onvif_backchannel_retries_digest_auth_from_config_credentials(
    digest_qop: str | None,
) -> None:
    # Given: A camera that requires Digest auth and credentials from adapter config
    server = _FakeRTSPBackchannelServer(require_digest_auth=True, digest_qop=digest_qop)
    await server.start()
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(
                rtsp_url=server.url,
                username="alice",
                password="s3cr@t",
            ),
            camera_name="front_door",
        )

        # When: Opening and closing a talk session
        session = await adapter.open_session(session_id="talk-1")
        await session.close()
    finally:
        await server.stop()

    # Then: The adapter retries DESCRIBE with a Digest authorization header
    assert [request.method for request in server.requests] == [
        "DESCRIBE",
        "DESCRIBE",
        "SETUP",
        "PLAY",
        "TEARDOWN",
    ]
    assert server.requests[1].headers["authorization"].startswith("Digest ")
    assert "@" not in server.requests[1].uri
    assert server.interleaved_before_play == []
    assert server.interleaved_after_play == []


@pytest.mark.asyncio
async def test_onvif_backchannel_fails_auth_when_credentials_are_missing() -> None:
    # Given: A camera that requires auth but the adapter has no credentials
    server = _FakeRTSPBackchannelServer(require_basic_auth=True)
    await server.start()
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=server.url),
            camera_name="front_door",
        )

        # When: Opening a talk session
        with pytest.raises(RTSPAuthenticationError, match="no credentials"):
            await adapter.open_session(session_id="talk-1")
    finally:
        await server.stop()

    # Then: The adapter fails after the challenged DESCRIBE without setup or RTP
    assert [request.method for request in server.requests] == ["DESCRIBE"]
    assert server.interleaved_before_play == []
    assert server.interleaved_after_play == []


@pytest.mark.asyncio
async def test_onvif_backchannel_fails_auth_when_camera_rejects_retry() -> None:
    # Given: A Basic-auth camera and wrong URL credentials
    server = _FakeRTSPBackchannelServer(require_basic_auth=True)
    await server.start()
    try:
        credentialed_url = server.url.replace("rtsp://", "rtsp://alice:wrong@", 1)
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=credentialed_url),
            camera_name="front_door",
        )

        # When: Opening a talk session
        with pytest.raises(RTSPAuthenticationError, match="rejected"):
            await adapter.open_session(session_id="talk-1")
    finally:
        await server.stop()

    # Then: The adapter retries once and maps the second challenge to auth failure
    assert [request.method for request in server.requests] == ["DESCRIBE", "DESCRIBE"]
    assert server.interleaved_before_play == []
    assert server.interleaved_after_play == []


@pytest.mark.asyncio
async def test_onvif_backchannel_fails_digest_auth_when_camera_rejects_retry() -> None:
    # Given: A Digest-auth camera and wrong configured credentials
    server = _FakeRTSPBackchannelServer(require_digest_auth=True)
    await server.start()
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(
                rtsp_url=server.url,
                username="alice",
                password="wrong",
            ),
            camera_name="front_door",
        )

        # When: Opening a talk session
        with pytest.raises(RTSPAuthenticationError, match="rejected"):
            await adapter.open_session(session_id="talk-1")
    finally:
        await server.stop()

    # Then: The adapter retries once and maps the second challenge to auth failure
    assert [request.method for request in server.requests] == ["DESCRIBE", "DESCRIBE"]
    assert server.interleaved_before_play == []
    assert server.interleaved_after_play == []


@pytest.mark.asyncio
async def test_onvif_backchannel_rejects_setup_failure_before_audio() -> None:
    # Given: A camera whose backchannel SETUP request fails
    server = _FakeRTSPBackchannelServer(setup_status=454)
    await server.start()
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=server.url),
            camera_name="front_door",
        )

        # When: Opening a talk session
        with pytest.raises(CameraRejectedTalkSessionError):
            await adapter.open_session(session_id="talk-1")
    finally:
        await server.stop()

    # Then: The adapter never reaches PLAY or sends RTP
    assert [request.method for request in server.requests] == ["DESCRIBE", "SETUP"]
    assert server.interleaved_before_play == []
    assert server.interleaved_after_play == []


@pytest.mark.asyncio
async def test_onvif_backchannel_rejects_setup_200_without_session_before_play() -> None:
    # Given: A camera that accepts SETUP without returning an RTSP session header
    server = _FakeRTSPBackchannelServer(setup_session_header=None)
    await server.start()
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=server.url),
            camera_name="front_door",
        )

        # When: Opening a talk session
        with pytest.raises(CameraRejectedTalkSessionError, match="Session"):
            await adapter.open_session(session_id="talk-1")
    finally:
        await server.stop()

    # Then: The adapter rejects the session before PLAY or RTP
    assert [request.method for request in server.requests] == ["DESCRIBE", "SETUP"]
    assert server.interleaved_before_play == []
    assert server.interleaved_after_play == []


@pytest.mark.asyncio
async def test_onvif_backchannel_rejects_play_failure_before_audio() -> None:
    # Given: A camera whose PLAY request fails after SETUP succeeds
    server = _FakeRTSPBackchannelServer(play_status=454)
    await server.start()
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=server.url),
            camera_name="front_door",
        )

        # When: Opening a talk session
        with pytest.raises(CameraRejectedTalkSessionError):
            await adapter.open_session(session_id="talk-1")
    finally:
        await server.stop()

    # Then: The adapter tears down the RTSP session and emits no RTP
    assert [request.method for request in server.requests] == [
        "DESCRIBE",
        "SETUP",
        "PLAY",
        "TEARDOWN",
    ]
    assert server.interleaved_before_play == []
    assert server.interleaved_after_play == []


@pytest.mark.asyncio
async def test_onvif_backchannel_cancelled_open_tears_down_setup_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: ONVIF SETUP has succeeded and PLAY is still waiting when open is cancelled.
    class _CancelDuringPlayClient:
        def __init__(self) -> None:
            self.session_id: str | None = None
            self.play_started = asyncio.Event()
            self.teardown_calls = 0
            self.close_calls = 0

        async def connect(self) -> None:
            return None

        async def describe(self, *, headers: dict[str, str] | None = None) -> RTSPResponse:
            _ = headers
            return RTSPResponse(
                status_code=200,
                reason="OK",
                body=_BACKCHANNEL_SDP.encode(),
            )

        async def setup_interleaved(
            self,
            control_url: str,
            *,
            rtp_channel: int,
            headers: dict[str, str] | None = None,
        ) -> RTSPResponse:
            _ = (control_url, rtp_channel, headers)
            self.session_id = "homesec-talk"
            return RTSPResponse(
                status_code=200,
                reason="OK",
                headers={"transport": "RTP/AVP/TCP;unicast;interleaved=0-1"},
            )

        async def play(self, *, headers: dict[str, str] | None = None) -> RTSPResponse:
            _ = headers
            self.play_started.set()
            await asyncio.Future()
            raise AssertionError("unreachable")

        async def teardown(self, *, headers: dict[str, str] | None = None) -> None:
            _ = headers
            self.teardown_calls += 1

        async def close(self) -> None:
            self.close_calls += 1

    client = _CancelDuringPlayClient()
    adapter = ONVIFBackchannelAdapter(
        ONVIFBackchannelConfig(rtsp_url="rtsp://camera.local/stream1"),
        camera_name="front_door",
    )
    monkeypatch.setattr(adapter, "_client", lambda: client)
    open_task = asyncio.create_task(adapter.open_session(session_id="talk-1"))
    await client.play_started.wait()

    # When: Runtime shutdown cancels the in-flight open before PLAY completes.
    open_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await open_task

    # Then: The already-created RTSP session is torn down and the socket is closed.
    assert client.teardown_calls == 1
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_onvif_session_rejects_writes_after_close() -> None:
    # Given: A talk session that has already been closed
    client = _NoopTalkClient()
    session = ONVIFTalkSession(
        session_id="talk-1",
        camera_name="front_door",
        client=client,  # type: ignore[arg-type]
        selected=_selected(),
    )

    await session.close()

    # When: Writing audio to the closed session
    with pytest.raises(CameraTalkStreamFailedError, match="closed"):
        await session.write_pcm_frame(_pcm_frame_16khz_20ms())

    # Then: The session rejects the write and closes the client only once
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_onvif_session_wraps_interleaved_send_failures() -> None:
    # Given: A talk session whose RTSP client fails interleaved RTP writes
    session = ONVIFTalkSession(
        session_id="talk-1",
        camera_name="front_door",
        client=_FailingSendTalkClient(),  # type: ignore[arg-type]
        selected=_selected(),
    )

    # When: Writing one PCM frame
    # Then: The raw send failure is reported as a camera talk stream failure
    with pytest.raises(CameraTalkStreamFailedError, match="broken pipe"):
        await session.write_pcm_frame(_pcm_frame_16khz_20ms())


@pytest.mark.asyncio
async def test_onvif_session_encodes_selected_pcma_codec() -> None:
    # Given: An active talk session whose negotiated camera codec is PCMA.
    client = _NoopTalkClient()
    session = ONVIFTalkSession(
        session_id="talk-1",
        camera_name="front_door",
        client=client,  # type: ignore[arg-type]
        selected=_selected(codec_name="PCMA", payload_type=8),
    )
    pcm_frame = _pcm_frame_16khz_20ms()

    # When: The camera-selected backchannel codec is PCMA.
    await session.write_pcm_frame(pcm_frame)

    # Then: The RTP payload uses payload type 8 and G.711 A-law bytes.
    assert len(client.sent) == 1
    channel, packet = client.sent[0]
    assert channel == 0
    assert parse_rtp_header(packet)["payload_type"] == 8
    assert packet[12:] == encode_pcma(
        resample_pcm_s16le_mono(pcm_frame, input_rate=16000, output_rate=8000)
    )


@pytest.mark.asyncio
async def test_onvif_session_rejects_selected_codecs_without_encoder() -> None:
    # Given: An active talk session whose selected codec has no encoder.
    session = ONVIFTalkSession(
        session_id="talk-1",
        camera_name="front_door",
        client=_NoopTalkClient(),  # type: ignore[arg-type]
        selected=_selected(codec_name="OPUS", payload_type=111),
    )

    # When: Writing one PCM frame through the unsupported codec session
    # Then: The session fails before any RTP is emitted
    with pytest.raises(UnsupportedTalkCodecError, match="OPUS/8000"):
        await session.write_pcm_frame(_pcm_frame_16khz_20ms())


@pytest.mark.asyncio
async def test_onvif_session_close_is_idempotent_and_teardown_is_best_effort() -> None:
    # Given: A talk session whose TEARDOWN request fails during close
    client = _FailingTeardownTalkClient(session_id="homesec-talk")
    session = ONVIFTalkSession(
        session_id="talk-1",
        camera_name="front_door",
        client=client,  # type: ignore[arg-type]
        selected=_selected(),
    )

    # When: Closing the session more than once
    await session.close()
    await session.close()

    # Then: TEARDOWN is attempted once and client close remains idempotent
    assert client.teardown_calls == 1
    assert client.close_calls == 1
