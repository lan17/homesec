from __future__ import annotations

import asyncio
import base64
import struct
from dataclasses import dataclass, field

import pytest

from homesec.sources.rtsp.talk.errors import (
    CameraBackchannelUnsupportedError,
    CameraRejectedTalkSessionError,
)
from homesec.sources.rtsp.talk.models import ONVIFBackchannelConfig
from homesec.sources.rtsp.talk.onvif_backchannel import ONVIFBackchannelAdapter
from homesec.sources.rtsp.talk.rtp import parse_rtp_header

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
    require_basic_auth: bool = False
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

    def _response_for(self, request: _RTSPRequest) -> bytes:
        cseq = request.headers.get("cseq", "1")
        if request.method == "DESCRIBE":
            if self.require_basic_auth and "authorization" not in request.headers:
                return _rtsp_response(
                    cseq=cseq,
                    status=401,
                    reason="Unauthorized",
                    headers={"WWW-Authenticate": 'Basic realm="homesec-fake"'},
                )
            if self.describe_status != 200:
                return _rtsp_response(
                    cseq=cseq, status=self.describe_status, reason="Option Not Supported"
                )
            return _rtsp_response(
                cseq=cseq,
                headers={"Content-Type": "application/sdp"},
                body=_BACKCHANNEL_SDP.encode(),
            )
        if request.method == "SETUP":
            if self.setup_status != 200:
                return _rtsp_response(cseq=cseq, status=self.setup_status, reason="Setup Failed")
            return _rtsp_response(
                cseq=cseq,
                headers={
                    "Transport": "RTP/AVP/TCP;unicast;interleaved=0-1",
                    "Session": "homesec-talk;timeout=60",
                },
            )
        if request.method == "PLAY":
            if self.play_status == 200:
                return _rtsp_response(cseq=cseq, headers={"Session": "homesec-talk"})
            return _rtsp_response(cseq=cseq, status=self.play_status, reason="Play Failed")
        if request.method == "TEARDOWN":
            return _rtsp_response(cseq=cseq, headers={"Session": "homesec-talk"})
        return _rtsp_response(cseq=cseq, status=405, reason="Method Not Allowed")


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


def _pcm_frame_16khz_20ms() -> bytes:
    samples = [0, 1200] * 160
    return struct.pack(f"<{len(samples)}h", *samples)


@pytest.mark.asyncio
async def test_onvif_backchannel_sends_no_rtp_before_play_200_ok() -> None:
    server = _FakeRTSPBackchannelServer()
    await server.start()
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=server.url),
            camera_name="front_door",
        )
        session = await adapter.open_session(session_id="talk-1")
        await session.write_pcm_frame(_pcm_frame_16khz_20ms())
        await session.close()
    finally:
        await server.stop()

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
async def test_onvif_backchannel_maps_describe_551_to_unsupported() -> None:
    server = _FakeRTSPBackchannelServer(describe_status=551)
    await server.start()
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=server.url),
            camera_name="front_door",
        )
        with pytest.raises(CameraBackchannelUnsupportedError):
            await adapter.open_session(session_id="talk-1")
    finally:
        await server.stop()

    assert [request.method for request in server.requests] == ["DESCRIBE"]
    assert server.interleaved_before_play == []
    assert server.interleaved_after_play == []


@pytest.mark.asyncio
async def test_onvif_backchannel_retries_basic_auth_from_url_credentials() -> None:
    server = _FakeRTSPBackchannelServer(require_basic_auth=True)
    await server.start()
    try:
        credentialed_url = server.url.replace("rtsp://", "rtsp://alice:s3cr%40t@", 1)
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=credentialed_url),
            camera_name="front_door",
        )
        session = await adapter.open_session(session_id="talk-1")
        await session.close()
    finally:
        await server.stop()

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


@pytest.mark.asyncio
async def test_onvif_backchannel_rejects_setup_failure_before_audio() -> None:
    server = _FakeRTSPBackchannelServer(setup_status=454)
    await server.start()
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=server.url),
            camera_name="front_door",
        )
        with pytest.raises(CameraRejectedTalkSessionError):
            await adapter.open_session(session_id="talk-1")
    finally:
        await server.stop()

    assert [request.method for request in server.requests] == ["DESCRIBE", "SETUP"]
    assert server.interleaved_before_play == []
    assert server.interleaved_after_play == []


@pytest.mark.asyncio
async def test_onvif_backchannel_rejects_play_failure_before_audio() -> None:
    server = _FakeRTSPBackchannelServer(play_status=454)
    await server.start()
    try:
        adapter = ONVIFBackchannelAdapter(
            ONVIFBackchannelConfig(rtsp_url=server.url),
            camera_name="front_door",
        )
        with pytest.raises(CameraRejectedTalkSessionError):
            await adapter.open_session(session_id="talk-1")
    finally:
        await server.stop()

    assert [request.method for request in server.requests] == ["DESCRIBE", "SETUP", "PLAY"]
    assert server.interleaved_before_play == []
    assert server.interleaved_after_play == []
