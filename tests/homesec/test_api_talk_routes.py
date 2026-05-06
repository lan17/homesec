"""Tests for push-to-talk control-plane and WebSocket API routes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from homesec.api.server import create_app
from homesec.api.talk_tokens import validate_camera_talk_token
from homesec.models.config import FastAPIServerConfig, TalkConfig
from homesec.models.talk import CameraTalkStatus, TalkInputFormat, TalkRefusalReason, TalkState
from homesec.runtime.errors import (
    TalkCameraNotFoundError,
    TalkRuntimeUnavailableError,
    TalkStreamOpenRefused,
)
from homesec.runtime.models import (
    CameraTalkSessionPrepared,
    CameraTalkStartRefusal,
    CameraTalkStopResult,
    RuntimeTalkStream,
)
from tests.homesec.ui_dist_stub import ensure_stub_ui_dist


class _MemoryTalkWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = False
        self.drain_count = 0

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    async def drain(self) -> None:
        self.drain_count += 1

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        self.closed = True


class _StubTalkApp:
    def __init__(
        self,
        *,
        status: CameraTalkStatus | None = None,
        prepare_result: CameraTalkSessionPrepared | CameraTalkStartRefusal | None = None,
        stop_result: CameraTalkStopResult | None = None,
        server_config: FastAPIServerConfig | None = None,
        talk_config: TalkConfig | None = None,
        bootstrap_mode: bool = False,
        status_error: Exception | None = None,
        prepare_error: Exception | None = None,
        open_error: Exception | None = None,
        stop_error: Exception | None = None,
        stream_writer: _MemoryTalkWriter | None = None,
    ) -> None:
        resolved_server = ensure_stub_ui_dist(server_config or FastAPIServerConfig())
        resolved_talk = talk_config or TalkConfig(enabled=True)
        self._bootstrap_mode = bootstrap_mode
        self._status = status or CameraTalkStatus(
            camera_name="front",
            enabled=True,
            state=TalkState.IDLE,
            supported_codecs=["pcmu"],
        )
        self._prepare_result = prepare_result or CameraTalkSessionPrepared(
            camera_name="front",
            session_id="session-1",
            input=resolved_talk.input,
        )
        self._stop_result = stop_result or CameraTalkStopResult(
            camera_name="front",
            accepted=True,
            state=TalkState.STOPPING,
        )
        self._status_error = status_error
        self._prepare_error = prepare_error
        self._open_error = open_error
        self._stop_error = stop_error
        self.stream_writer = stream_writer or _MemoryTalkWriter()
        self.prepare_calls: list[tuple[str, str, TalkInputFormat]] = []
        self.open_calls: list[tuple[str, str, TalkInputFormat]] = []
        self.stop_calls: list[tuple[str, str]] = []
        self._config = SimpleNamespace(
            server=resolved_server,
            talk=resolved_talk,
            cameras=[
                SimpleNamespace(
                    name="front",
                    enabled=True,
                    source=SimpleNamespace(backend="rtsp"),
                )
            ],
        )

    @property
    def config(self):
        return self._config

    @property
    def server_config(self) -> FastAPIServerConfig:
        return self._config.server

    @property
    def bootstrap_mode(self) -> bool:
        return self._bootstrap_mode

    async def get_camera_talk_status(self, camera_name: str) -> CameraTalkStatus:
        if self._status_error is not None:
            raise self._status_error
        assert camera_name == self._status.camera_name
        return self._status

    async def prepare_camera_talk_session(
        self,
        camera_name: str,
        *,
        session_id: str,
        input_format: TalkInputFormat,
    ) -> CameraTalkSessionPrepared | CameraTalkStartRefusal:
        if self._prepare_error is not None:
            raise self._prepare_error
        self.prepare_calls.append((camera_name, session_id, input_format))
        return self._prepare_result

    async def open_camera_talk_stream(
        self,
        camera_name: str,
        *,
        session_id: str,
        input_format: TalkInputFormat,
    ) -> RuntimeTalkStream:
        if self._open_error is not None:
            raise self._open_error
        self.open_calls.append((camera_name, session_id, input_format))
        return RuntimeTalkStream(
            camera_name=camera_name,
            session_id=session_id,
            input=input_format,
            reader=cast(asyncio.StreamReader, object()),
            writer=cast(asyncio.StreamWriter, self.stream_writer),
        )

    async def stop_camera_talk_session(
        self,
        camera_name: str,
        *,
        session_id: str,
    ) -> CameraTalkStopResult:
        if self._stop_error is not None:
            raise self._stop_error
        self.stop_calls.append((camera_name, session_id))
        return self._stop_result


def _client(app: _StubTalkApp) -> TestClient:
    return TestClient(create_app(app))


def test_get_talk_status_returns_runtime_status() -> None:
    """GET /talk/cameras/{camera_name} should mirror runtime talk status."""
    app = _StubTalkApp(
        status=CameraTalkStatus(
            camera_name="front",
            enabled=True,
            state=TalkState.ACTIVE,
            active_session_id="session-1",
            supported_codecs=["pcmu"],
            selected_codec="pcmu",
            last_error=None,
        )
    )
    client = _client(app)

    response = client.get("/api/v1/talk/cameras/front")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "camera_name": "front",
        "enabled": True,
        "state": "active",
        "active_session_id": "session-1",
        "supported_codecs": ["pcmu"],
        "selected_codec": "pcmu",
        "last_error": None,
    }


def test_prepare_talk_session_returns_stream_url_and_runtime_contract() -> None:
    """POST /talk/cameras/{camera_name}/sessions should reserve and describe a stream."""
    input_format = TalkInputFormat(sample_rate=8000, frame_ms=20)
    app = _StubTalkApp(
        talk_config=TalkConfig(enabled=True, max_session_s=45, idle_timeout_s=1.5),
        prepare_result=CameraTalkSessionPrepared(
            camera_name="front",
            session_id="custom-session",
            input=input_format,
        ),
    )
    client = _client(app)

    response = client.post(
        "/api/v1/talk/cameras/front/sessions",
        json={"session_id": "custom-session", "input": input_format.model_dump(mode="json")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["camera_name"] == "front"
    assert payload["session_id"] == "custom-session"
    assert payload["state"] == "starting"
    assert payload["input"] == input_format.model_dump(mode="json")
    assert payload["token"] is None
    assert payload["token_expires_at"] is None
    assert payload["max_session_s"] == 45
    assert payload["idle_timeout_s"] == 1.5
    parsed = urlparse(payload["stream_url"])
    assert parsed.path == "/api/v1/talk/cameras/front/sessions/custom-session/stream"
    assert parse_qs(parsed.query) == {
        "codec": ["pcm_s16le"],
        "sample_rate": ["8000"],
        "channels": ["1"],
        "frame_ms": ["20"],
    }
    assert app.prepare_calls == [("front", "custom-session", input_format)]


def test_prepare_talk_session_mints_short_lived_stream_token_when_auth_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authenticated deployments should return a session-scoped WebSocket token."""
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    app = _StubTalkApp(
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY"),
        talk_config=TalkConfig(enabled=True, token_ttl_s=42),
        prepare_result=CameraTalkSessionPrepared(
            camera_name="front",
            session_id="session-1",
            input=TalkInputFormat(),
        ),
    )
    client = _client(app)

    response = client.post(
        "/api/v1/talk/cameras/front/sessions",
        headers={"Authorization": "Bearer secret"},
        json={"session_id": "session-1"},
    )

    assert response.status_code == 201
    payload = response.json()
    token = payload["token"]
    assert isinstance(token, str)
    assert payload["token_expires_at"] is not None
    assert parse_qs(urlparse(payload["stream_url"]).query)["token"] == [token]
    decoded = validate_camera_talk_token(
        api_key="secret",
        token=token,
        camera_name="front",
        session_id="session-1",
    )
    assert decoded.camera_name == "front"
    assert decoded.session_id == "session-1"


def test_talk_control_routes_require_api_key_when_auth_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Talk control endpoints should inherit normal API-key auth."""
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    app = _StubTalkApp(
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY")
    )
    client = _client(app)

    response = client.get("/api/v1/talk/cameras/front")

    assert response.status_code == 401
    assert response.json()["error_code"] == "UNAUTHORIZED"


def test_prepare_talk_session_maps_runtime_refusal_to_api_error() -> None:
    """Machine-readable talk refusal reasons should survive the HTTP boundary."""
    app = _StubTalkApp(
        prepare_result=CameraTalkStartRefusal(
            camera_name="front",
            reason=TalkRefusalReason.SESSION_ALREADY_ACTIVE,
            message="A talk session is already active for this camera",
        )
    )
    client = _client(app)

    response = client.post("/api/v1/talk/cameras/front/sessions", json={"session_id": "session-1"})

    assert response.status_code == 409
    assert response.json() == {
        "detail": "A talk session is already active for this camera",
        "error_code": "TALK_SESSION_ALREADY_ACTIVE",
        "reason": "session_already_active",
    }


def test_talk_routes_map_camera_not_found() -> None:
    """Unknown runtime cameras should return the talk-specific 404 envelope."""
    app = _StubTalkApp(status_error=TalkCameraNotFoundError("missing"))
    client = _client(app)

    response = client.get("/api/v1/talk/cameras/missing")

    assert response.status_code == 404
    assert response.json()["error_code"] == "TALK_CAMERA_NOT_FOUND"


def test_delete_talk_session_returns_stop_result() -> None:
    """DELETE /talk/cameras/{camera_name}/sessions/{session_id} should stop runtime talk."""
    app = _StubTalkApp(
        stop_result=CameraTalkStopResult(
            camera_name="front",
            accepted=True,
            state=TalkState.STOPPING,
        )
    )
    client = _client(app)

    response = client.delete("/api/v1/talk/cameras/front/sessions/session-1")

    assert response.status_code == 202
    assert response.json() == {"accepted": True, "state": "stopping"}
    assert app.stop_calls == [("front", "session-1")]


def test_talk_websocket_forwards_binary_frames_to_runtime_stream() -> None:
    """The WebSocket should bridge exact-size PCM frames into length-prefixed IPC."""
    input_format = TalkInputFormat(sample_rate=8000, frame_ms=10)
    writer = _MemoryTalkWriter()
    app = _StubTalkApp(
        talk_config=TalkConfig(enabled=True, input=input_format),
        stream_writer=writer,
    )
    client = _client(app)
    frame = b"\x11" * input_format.expected_bytes_per_frame

    with client.websocket_connect(
        "/api/v1/talk/cameras/front/sessions/session-1/stream"
        "?codec=pcm_s16le&sample_rate=8000&channels=1&frame_ms=10"
    ) as websocket:
        assert websocket.receive_json() == {
            "type": "ready",
            "camera_name": "front",
            "session_id": "session-1",
            "input": input_format.model_dump(mode="json"),
        }
        websocket.send_bytes(frame)
        websocket.send_text("stop")
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_text()

    assert app.open_calls == [("front", "session-1", input_format)]
    assert app.stop_calls == [("front", "session-1")]
    assert writer.closed is True
    assert writer.drain_count == 1
    assert bytes(writer.data[:4]) == len(frame).to_bytes(4, "big")
    assert bytes(writer.data[4:]) == frame


def test_talk_websocket_forwards_multiple_binary_frames_to_runtime_stream() -> None:
    """The WebSocket bridge should stay open for the whole talk stream."""
    input_format = TalkInputFormat(sample_rate=8000, frame_ms=10)
    writer = _MemoryTalkWriter()
    app = _StubTalkApp(
        talk_config=TalkConfig(enabled=True, input=input_format),
        stream_writer=writer,
    )
    client = _client(app)
    first_frame = b"\x11" * input_format.expected_bytes_per_frame
    second_frame = b"\x22" * input_format.expected_bytes_per_frame

    with client.websocket_connect(
        "/api/v1/talk/cameras/front/sessions/session-1/stream"
        "?codec=pcm_s16le&sample_rate=8000&channels=1&frame_ms=10"
    ) as websocket:
        websocket.receive_json()
        websocket.send_bytes(first_frame)
        websocket.send_bytes(second_frame)
        websocket.send_text("stop")
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_text()

    frame_len = input_format.expected_bytes_per_frame
    assert writer.drain_count == 2
    assert bytes(writer.data[:4]) == frame_len.to_bytes(4, "big")
    assert bytes(writer.data[4 : 4 + frame_len]) == first_frame
    second_offset = 4 + frame_len
    assert bytes(writer.data[second_offset : second_offset + 4]) == frame_len.to_bytes(4, "big")
    assert bytes(writer.data[second_offset + 4 :]) == second_frame


def test_talk_websocket_rejects_invalid_audio_frame_length() -> None:
    """Invalid browser frame sizes should not be forwarded into runtime IPC."""
    input_format = TalkInputFormat(sample_rate=8000, frame_ms=10)
    writer = _MemoryTalkWriter()
    app = _StubTalkApp(
        talk_config=TalkConfig(enabled=True, input=input_format),
        stream_writer=writer,
    )
    client = _client(app)

    with client.websocket_connect(
        "/api/v1/talk/cameras/front/sessions/session-1/stream"
        "?codec=pcm_s16le&sample_rate=8000&channels=1&frame_ms=10"
    ) as websocket:
        websocket.receive_json()
        websocket.send_bytes(b"too short")
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_text()

    assert bytes(writer.data) == b""
    assert app.stop_calls == [("front", "session-1")]


def test_talk_websocket_cleans_up_reserved_session_on_invalid_input_query() -> None:
    """Invalid attach parameters should not leave a reserved talk slot behind."""
    app = _StubTalkApp()
    client = _client(app)

    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect(
            "/api/v1/talk/cameras/front/sessions/session-1/stream?sample_rate=not-an-int"
        ),
    ):
        pass

    assert app.open_calls == []
    assert app.stop_calls == [("front", "session-1")]


def test_talk_websocket_cleans_up_when_runtime_stream_open_fails() -> None:
    """Runtime restart/stale-socket failures during attach should stop the reservation."""
    app = _StubTalkApp(open_error=TalkRuntimeUnavailableError("stale runtime socket"))
    client = _client(app)

    with (
        client.websocket_connect(
            "/api/v1/talk/cameras/front/sessions/session-1/stream"
        ) as websocket,
        pytest.raises(WebSocketDisconnect),
    ):
        websocket.receive_text()

    assert app.open_calls == []
    assert app.stop_calls == [("front", "session-1")]


def test_talk_websocket_maps_typed_stream_open_refusal_to_policy_close() -> None:
    """Attach-time typed refusals should survive to WebSocket close semantics."""
    app = _StubTalkApp(
        open_error=TalkStreamOpenRefused(
            "Talk session is not reserved",
            reason=TalkRefusalReason.INVALID_AUDIO_FRAME,
        )
    )
    client = _client(app)

    with (
        client.websocket_connect(
            "/api/v1/talk/cameras/front/sessions/session-1/stream"
        ) as websocket,
        pytest.raises(WebSocketDisconnect) as disconnect,
    ):
        websocket.receive_text()

    assert disconnect.value.code == 1008
    assert disconnect.value.reason == "Talk stream refused"
    assert app.open_calls == []
    assert app.stop_calls == [("front", "session-1")]


def test_talk_websocket_accepts_short_lived_token_when_auth_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Browser WebSocket streams should authorize via the token returned by prepare."""
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    input_format = TalkInputFormat(sample_rate=8000, frame_ms=10)
    app = _StubTalkApp(
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY"),
        talk_config=TalkConfig(enabled=True, input=input_format),
        prepare_result=CameraTalkSessionPrepared(
            camera_name="front",
            session_id="session-1",
            input=input_format,
        ),
    )
    client = _client(app)
    prepare = client.post(
        "/api/v1/talk/cameras/front/sessions",
        headers={"Authorization": "Bearer secret"},
        json={"session_id": "session-1", "input": input_format.model_dump(mode="json")},
    )
    stream_url = prepare.json()["stream_url"]

    with client.websocket_connect(stream_url) as websocket:
        ready = websocket.receive_json()
        websocket.send_text("stop")
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_text()

    assert ready["type"] == "ready"
    assert app.open_calls == [("front", "session-1", input_format)]


def test_talk_websocket_rejects_missing_token_when_auth_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authenticated WebSocket streams should fail closed without a bearer key or stream token."""
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    app = _StubTalkApp(
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY")
    )
    client = _client(app)

    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/api/v1/talk/cameras/front/sessions/session-1/stream"),
    ):
        pass
    assert app.open_calls == []
