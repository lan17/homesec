"""Tests for push-to-talk control-plane and WebSocket API routes."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import homesec.api.routes.talk as talk_route_module
import homesec.api.talk_tokens as talk_tokens
from homesec.api.errors import APIErrorCode
from homesec.api.server import create_app
from homesec.api.talk_tokens import issue_camera_talk_token, validate_camera_talk_token
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
    def __init__(
        self,
        *,
        drain_error: Exception | None = None,
        wait_closed_error: Exception | None = None,
    ) -> None:
        self.data = bytearray()
        self.closed = False
        self.drain_count = 0
        self.drain_error = drain_error
        self.wait_closed_error = wait_closed_error

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    async def drain(self) -> None:
        self.drain_count += 1
        if self.drain_error is not None:
            raise self.drain_error

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        self.closed = True
        if self.wait_closed_error is not None:
            raise self.wait_closed_error


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
            selected_codec=self._status.selected_codec or "PCMU/8000",
            backend=self._status.backend,
            backend_reason=self._status.backend_reason,
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


def _start_message(input_format: TalkInputFormat) -> str:
    return json.dumps({"type": "start", **input_format.model_dump(mode="json")})


def _stop_message() -> str:
    return json.dumps({"type": "stop"})


def test_get_talk_status_returns_runtime_status() -> None:
    """GET /talk/cameras/{camera_name} should mirror runtime talk status."""
    # Given: The runtime reports an active talk session with negotiated status fields
    app = _StubTalkApp(
        status=CameraTalkStatus(
            camera_name="front",
            enabled=True,
            state=TalkState.ACTIVE,
            active_session_id="session-1",
            supported_codecs=["pcmu"],
            selected_codec="pcmu",
            backend="onvif_rtsp_backchannel",
            backend_reason="Selected ONVIF RTSP backchannel by standards-first auto probing",
            last_error=None,
        )
    )
    client = _client(app)

    # When: Fetching talk status for the camera
    response = client.get("/api/v1/talk/cameras/front")

    # Then: The API mirrors the runtime status and compatibility fields
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "camera_name": "front",
        "enabled": True,
        "policy_enabled": True,
        "capability": "supported",
        "state": "active",
        "active_session_id": "session-1",
        "supported_codecs": ["pcmu"],
        "offered_codecs": [],
        "selected_codec": "pcmu",
        "backend": "onvif_rtsp_backchannel",
        "backend_reason": "Selected ONVIF RTSP backchannel by standards-first auto probing",
        "last_error": None,
    }


def test_get_talk_status_returns_disabled_camera_status() -> None:
    """Disabled talk status should be reported without opening a session."""
    # Given: The runtime reports talk disabled for the requested camera.
    # When: The client reads the camera talk status endpoint.
    # Then: The API returns the disabled status without opening a talk stream.
    app = _StubTalkApp(
        status=CameraTalkStatus(
            camera_name="front",
            enabled=False,
            state=TalkState.DISABLED,
            supported_codecs=[],
            selected_codec=None,
            last_error=None,
        )
    )
    client = _client(app)

    response = client.get("/api/v1/talk/cameras/front")

    assert response.status_code == 200
    assert response.json()["enabled"] is False
    assert response.json()["state"] == "disabled"
    assert response.json()["backend"] is None
    assert response.json()["backend_reason"] is None
    assert app.open_calls == []


def test_get_talk_status_returns_unsupported_source_status() -> None:
    """Unsupported sources should surface as talk status rather than session opens."""
    # Given: The runtime reports that the camera source has no talk support.
    # When: The client reads the camera talk status endpoint.
    # Then: The API returns unsupported state and preserves the diagnostic error.
    app = _StubTalkApp(
        status=CameraTalkStatus(
            camera_name="front",
            enabled=True,
            state=TalkState.UNSUPPORTED,
            supported_codecs=[],
            selected_codec=None,
            last_error="Source does not support talk",
        )
    )
    client = _client(app)

    response = client.get("/api/v1/talk/cameras/front")

    assert response.status_code == 200
    assert response.json()["state"] == "unsupported"
    assert response.json()["last_error"] == "Source does not support talk"
    assert app.open_calls == []


def test_prepare_talk_session_returns_websocket_url_and_runtime_contract() -> None:
    """POST /talk/cameras/{camera_name}/sessions should reserve and describe a stream."""
    # Given: Runtime talk preparation reserves a session with explicit input settings.
    # When: The client creates a talk session over the REST endpoint.
    # Then: The response exposes the session, stream URL, timing, and input contract.
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
    assert payload["websocket_url"] == payload["stream_url"]
    parsed = urlparse(payload["websocket_url"])
    assert parsed.path == "/api/v1/talk/cameras/front/sessions/custom-session/stream"
    assert parse_qs(parsed.query) == {}
    assert app.prepare_calls == [("front", "custom-session", input_format)]


def test_prepare_talk_session_uses_configured_input_when_request_omits_input() -> None:
    """POST /talk sessions should inherit the server-configured browser PCM contract."""
    configured_input = TalkInputFormat(sample_rate=8000, frame_ms=20)
    app = _StubTalkApp(
        talk_config=TalkConfig(enabled=True, input=configured_input),
        prepare_result=CameraTalkSessionPrepared(
            camera_name="front",
            session_id="session-1",
            input=configured_input,
        ),
    )
    client = _client(app)

    # Given: A server configured with a non-default talk input format
    # When: Preparing a session without an explicit input override
    response = client.post(
        "/api/v1/talk/cameras/front/sessions",
        json={"session_id": "session-1"},
    )

    # Then: The runtime prepare call and response use the configured input format
    assert response.status_code == 201
    assert app.prepare_calls == [("front", "session-1", configured_input)]
    assert response.json()["input"] == configured_input.model_dump(mode="json")


def test_prepare_talk_session_generates_session_id_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session creation should mint an opaque id when the client does not provide one."""
    # Given: The client omits a talk session id during preparation.
    # When: The REST endpoint creates the session.
    # Then: The API generates an opaque talk session id before calling runtime.
    monkeypatch.setattr(talk_route_module.secrets, "token_urlsafe", lambda size: f"fixed-{size}")
    app = _StubTalkApp()
    client = _client(app)

    response = client.post("/api/v1/talk/cameras/front/sessions", json={})

    assert response.status_code == 201
    generated_session_id = app.prepare_calls[0][1]
    assert generated_session_id == "tk_fixed-16"


def test_prepare_talk_session_mints_short_lived_stream_token_when_auth_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authenticated deployments should return a session-scoped WebSocket token."""
    # Given: API authentication is enabled and talk tokens have a short TTL.
    # When: An authenticated client prepares a talk session.
    # Then: The response includes a session-scoped WebSocket token and URL.
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
    assert payload["websocket_url"] == payload["stream_url"]
    assert parse_qs(urlparse(payload["websocket_url"]).query)["token"] == [token]
    decoded = validate_camera_talk_token(
        api_key="secret",
        token=token,
        camera_name="front",
        session_id="session-1",
    )
    assert decoded.camera_name == "front"
    assert decoded.session_id == "session-1"


def test_prepare_talk_session_reports_missing_api_key_when_auth_enabled() -> None:
    """Auth-enabled session creation should fail closed if no API key is configured."""
    # Given: API authentication is enabled but the configured API key env var is absent.
    # When: A client attempts to create a talk session.
    # Then: The API fails closed with the missing-key error envelope.
    app = _StubTalkApp(
        server_config=FastAPIServerConfig(
            auth_enabled=True,
            api_key_env="HOMESEC_MISSING_TEST_API_KEY",
        )
    )
    client = _client(app)

    response = client.post(
        "/api/v1/talk/cameras/front/sessions",
        headers={"Authorization": "Bearer anything"},
        json={"session_id": "session-1"},
    )

    assert response.status_code == 500
    assert response.json()["error_code"] == APIErrorCode.API_KEY_NOT_CONFIGURED


def test_talk_control_routes_require_api_key_when_auth_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Talk control endpoints should inherit normal API-key auth."""
    # Given: API authentication is enabled for talk control routes.
    # When: A client calls talk status without credentials.
    # Then: The route rejects the request with normal unauthorized semantics.
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    app = _StubTalkApp(
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY")
    )
    client = _client(app)

    response = client.get("/api/v1/talk/cameras/front")

    assert response.status_code == 401
    assert response.json()["error_code"] == "UNAUTHORIZED"


def test_prepare_talk_session_requires_api_key_when_auth_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session-creation route should require normal API authentication."""
    # Given: API authentication is enabled for talk session creation.
    # When: A client prepares a session without credentials.
    # Then: The API rejects the request before reserving runtime talk.
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    app = _StubTalkApp(
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY")
    )
    client = _client(app)

    response = client.post("/api/v1/talk/cameras/front/sessions", json={"session_id": "s1"})

    assert response.status_code == 401
    assert response.json()["error_code"] == "UNAUTHORIZED"
    assert app.prepare_calls == []


@pytest.mark.parametrize(
    ("reason", "expected_status", "expected_error_code"),
    [
        (
            TalkRefusalReason.CAMERA_NOT_FOUND,
            status.HTTP_404_NOT_FOUND,
            APIErrorCode.TALK_CAMERA_NOT_FOUND,
        ),
        (TalkRefusalReason.TALK_DISABLED, status.HTTP_409_CONFLICT, APIErrorCode.TALK_DISABLED),
        (
            TalkRefusalReason.SOURCE_NOT_TALK_CAPABLE,
            status.HTTP_409_CONFLICT,
            APIErrorCode.TALK_SOURCE_NOT_TALK_CAPABLE,
        ),
        (
            TalkRefusalReason.SESSION_ALREADY_ACTIVE,
            status.HTTP_409_CONFLICT,
            APIErrorCode.TALK_SESSION_ALREADY_ACTIVE,
        ),
        (
            TalkRefusalReason.SESSION_BUDGET_EXHAUSTED,
            status.HTTP_409_CONFLICT,
            APIErrorCode.TALK_SESSION_BUDGET_EXHAUSTED,
        ),
        (
            TalkRefusalReason.UNSUPPORTED_CAMERA,
            status.HTTP_409_CONFLICT,
            APIErrorCode.TALK_UNSUPPORTED_CAMERA,
        ),
        (
            TalkRefusalReason.UNSUPPORTED_CODEC,
            status.HTTP_400_BAD_REQUEST,
            APIErrorCode.TALK_UNSUPPORTED_CODEC,
        ),
        (
            TalkRefusalReason.CAMERA_BACKCHANNEL_FAILED,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            APIErrorCode.TALK_CAMERA_BACKCHANNEL_FAILED,
        ),
        (
            TalkRefusalReason.RUNTIME_UNAVAILABLE,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            APIErrorCode.TALK_RUNTIME_UNAVAILABLE,
        ),
        (
            TalkRefusalReason.INVALID_AUDIO_FRAME,
            status.HTTP_400_BAD_REQUEST,
            APIErrorCode.TALK_INVALID_AUDIO_FRAME,
        ),
        (
            TalkRefusalReason.BACKPRESSURE,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            APIErrorCode.TALK_BACKPRESSURE,
        ),
    ],
)
def test_prepare_talk_session_maps_runtime_refusal_to_api_error(
    reason: TalkRefusalReason,
    expected_status: int,
    expected_error_code: APIErrorCode,
) -> None:
    """Machine-readable talk refusal reasons should survive the HTTP boundary."""
    # Given: Runtime refuses talk preparation with a typed refusal reason.
    # When: The client creates a talk session through the REST endpoint.
    # Then: The API maps the refusal to the expected HTTP status and reason payload.
    app = _StubTalkApp(
        prepare_result=CameraTalkStartRefusal(
            camera_name="front",
            reason=reason,
            message="Talk session refused",
        )
    )
    client = _client(app)

    response = client.post("/api/v1/talk/cameras/front/sessions", json={"session_id": "session-1"})

    assert response.status_code == expected_status
    assert response.json() == {
        "detail": "Talk session refused",
        "error_code": expected_error_code,
        "reason": reason.value,
    }


def test_talk_routes_map_camera_not_found() -> None:
    """Unknown runtime cameras should return the talk-specific 404 envelope."""
    # Given: Runtime cannot find the requested camera during talk status lookup.
    # When: The client reads talk status for that camera.
    # Then: The API returns the talk-specific camera-not-found envelope.
    app = _StubTalkApp(status_error=TalkCameraNotFoundError("missing"))
    client = _client(app)

    response = client.get("/api/v1/talk/cameras/missing")

    assert response.status_code == 404
    assert response.json()["error_code"] == "TALK_CAMERA_NOT_FOUND"


@pytest.mark.parametrize("method", ["get", "post", "delete"])
def test_talk_control_routes_map_runtime_unavailable(method: str) -> None:
    """Control routes should expose runtime availability failures consistently."""
    # Given: Runtime talk control operations fail because the worker is unavailable.
    # When: The client calls a talk status, prepare, or stop route.
    # Then: The API reports a retryable runtime-unavailable error consistently.
    error = TalkRuntimeUnavailableError("worker unavailable")
    app = _StubTalkApp(
        status_error=error if method == "get" else None,
        prepare_error=error if method == "post" else None,
        stop_error=error if method == "delete" else None,
    )
    client = _client(app)

    if method == "get":
        response = client.get("/api/v1/talk/cameras/front")
    elif method == "post":
        response = client.post("/api/v1/talk/cameras/front/sessions", json={"session_id": "s1"})
    else:
        response = client.delete("/api/v1/talk/cameras/front/sessions/s1")

    assert response.status_code == 503
    assert response.json()["error_code"] == APIErrorCode.TALK_RUNTIME_UNAVAILABLE


@pytest.mark.parametrize("method", ["post", "delete"])
def test_talk_mutation_routes_map_camera_not_found(method: str) -> None:
    """Session mutation routes should keep camera-not-found errors talk-specific."""
    # Given: Runtime cannot find the requested camera for a talk mutation.
    # When: The client prepares or stops a talk session for that camera.
    # Then: The API preserves the talk-specific camera-not-found response.
    error = TalkCameraNotFoundError("missing")
    app = _StubTalkApp(
        prepare_error=error if method == "post" else None,
        stop_error=error if method == "delete" else None,
    )
    client = _client(app)

    if method == "post":
        response = client.post("/api/v1/talk/cameras/missing/sessions", json={"session_id": "s1"})
    else:
        response = client.delete("/api/v1/talk/cameras/missing/sessions/s1")

    assert response.status_code == 404
    assert response.json()["error_code"] == APIErrorCode.TALK_CAMERA_NOT_FOUND


def test_delete_talk_session_returns_stop_result() -> None:
    """DELETE /talk/cameras/{camera_name}/sessions/{session_id} should stop runtime talk."""
    # Given: Runtime accepts a request to stop an active talk session.
    # When: The client deletes the talk session resource.
    # Then: The API returns the stop result and forwards the session id to runtime.
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
    # Given: A prepared talk WebSocket and an exact-size browser PCM frame.
    # When: The client starts the stream, sends audio, and stops.
    # Then: The API writes one length-prefixed frame and closes runtime cleanly.
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
    ) as websocket:
        websocket.send_text(_start_message(input_format))
        assert websocket.receive_json() == {
            "type": "ready",
            "camera_name": "front",
            "session_id": "session-1",
            "input": input_format.model_dump(mode="json"),
            "camera_codec": "PCMU/8000",
            "backend": None,
            "backend_reason": None,
        }
        websocket.send_bytes(frame)
        websocket.send_text(_stop_message())
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
    # Given: A prepared talk WebSocket with two valid browser PCM frames.
    # When: The client sends both frames before stopping.
    # Then: The API forwards each frame in order with independent length prefixes.
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
    ) as websocket:
        websocket.send_text(_start_message(input_format))
        websocket.receive_json()
        websocket.send_bytes(first_frame)
        websocket.send_bytes(second_frame)
        websocket.send_text(_stop_message())
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_text()

    frame_len = input_format.expected_bytes_per_frame
    assert writer.drain_count == 2
    assert bytes(writer.data[:4]) == frame_len.to_bytes(4, "big")
    assert bytes(writer.data[4 : 4 + frame_len]) == first_frame
    second_offset = 4 + frame_len
    assert bytes(writer.data[second_offset : second_offset + 4]) == frame_len.to_bytes(4, "big")
    assert bytes(writer.data[second_offset + 4 :]) == second_frame


def test_talk_websocket_idle_timeout_closes_stream_when_browser_audio_stalls() -> None:
    """A ready WebSocket with no browser audio should not hold the runtime stream forever."""
    input_format = TalkInputFormat(sample_rate=8000, frame_ms=10)
    writer = _MemoryTalkWriter()
    app = _StubTalkApp(
        talk_config=TalkConfig(enabled=True, input=input_format, idle_timeout_s=0.1),
        stream_writer=writer,
    )
    client = _client(app)

    # Given: A talk WebSocket that reaches ready but receives no audio frames
    with client.websocket_connect(
        "/api/v1/talk/cameras/front/sessions/session-1/stream"
    ) as websocket:
        websocket.send_text(_start_message(input_format))
        websocket.receive_json()

        # When: The browser audio graph stalls past the configured idle timeout
        with pytest.raises(WebSocketDisconnect) as disconnect:
            websocket.receive_text()

    # Then: The API closes the stream and releases the reserved runtime session
    assert disconnect.value.code == 1000
    assert app.stop_calls == [("front", "session-1")]
    assert writer.closed is True
    assert bytes(writer.data) == b""


def test_talk_websocket_rejects_invalid_audio_frame_length() -> None:
    """Invalid browser frame sizes should not be forwarded into runtime IPC."""
    # Given: A prepared talk WebSocket receives a binary frame with the wrong size.
    # When: The client sends that invalid audio frame after start.
    # Then: The API closes the stream without forwarding bytes to runtime.
    input_format = TalkInputFormat(sample_rate=8000, frame_ms=10)
    writer = _MemoryTalkWriter()
    app = _StubTalkApp(
        talk_config=TalkConfig(enabled=True, input=input_format),
        stream_writer=writer,
    )
    client = _client(app)

    with client.websocket_connect(
        "/api/v1/talk/cameras/front/sessions/session-1/stream"
    ) as websocket:
        websocket.send_text(_start_message(input_format))
        websocket.receive_json()
        websocket.send_bytes(b"too short")
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_text()

    assert bytes(writer.data) == b""
    assert app.stop_calls == [("front", "session-1")]


def test_talk_websocket_cleans_up_reserved_session_on_invalid_start_message() -> None:
    """Invalid start control messages should not leave a reserved talk slot behind."""
    # Given: A prepared talk WebSocket receives a malformed start control message.
    # When: The client sends the invalid start payload.
    # Then: The API closes the socket and releases the reserved talk session.
    app = _StubTalkApp()
    client = _client(app)

    with client.websocket_connect(
        "/api/v1/talk/cameras/front/sessions/session-1/stream"
    ) as websocket:
        websocket.send_text(json.dumps({"type": "start", "sample_rate": "not-an-int"}))
        with pytest.raises(WebSocketDisconnect) as disconnect:
            websocket.receive_text()

    assert disconnect.value.code == 1008
    assert app.open_calls == []
    assert app.stop_calls == [("front", "session-1")]


def test_talk_websocket_disconnect_before_start_releases_reserved_session() -> None:
    """A browser closing before start should release the prepared talk slot."""
    app = _StubTalkApp()
    client = _client(app)

    # Given: A prepared talk WebSocket that is accepted by the API
    # When: The browser disconnects before sending the required start control message
    with client.websocket_connect("/api/v1/talk/cameras/front/sessions/session-1/stream"):
        pass

    # Then: The reservation is stopped without opening the runtime audio stream
    assert app.open_calls == []
    assert app.stop_calls == [("front", "session-1")]


def test_talk_websocket_start_timeout_releases_reserved_session() -> None:
    """A silent client should not hold a prepared talk slot forever."""
    app = _StubTalkApp(talk_config=TalkConfig(enabled=True, idle_timeout_s=0.1))
    client = _client(app)

    # Given: A prepared talk WebSocket is accepted but sends no start message.
    with (
        client.websocket_connect(
            "/api/v1/talk/cameras/front/sessions/session-1/stream"
        ) as websocket,
        # When: The start-message wait exceeds the configured talk idle timeout.
        pytest.raises(WebSocketDisconnect) as disconnect,
    ):
        websocket.receive_text()

    # Then: The API closes the socket and releases the reservation without opening runtime talk.
    assert disconnect.value.code == 1000
    assert disconnect.value.reason == "Talk start timeout"
    assert app.open_calls == []
    assert app.stop_calls == [("front", "session-1")]


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (b"not json", 1003),
        ("not json", 1008),
        (json.dumps(["not", "an", "object"]), 1008),
        (json.dumps({"type": "start"}), 1008),
        (json.dumps({"type": "start", "unexpected": True}), 1008),
    ],
)
def test_talk_websocket_rejects_invalid_start_payload_shapes(
    payload: str | bytes,
    expected_code: int,
) -> None:
    """The first WebSocket message must be a strict JSON start control object."""
    # Given: A prepared talk WebSocket receives an invalid first-message shape.
    # When: The client sends the invalid start payload variant.
    # Then: The API closes with the expected policy/protocol code and releases runtime.
    app = _StubTalkApp()
    client = _client(app)

    with client.websocket_connect(
        "/api/v1/talk/cameras/front/sessions/session-1/stream"
    ) as websocket:
        if isinstance(payload, bytes):
            websocket.send_bytes(payload)
        else:
            websocket.send_text(payload)
        with pytest.raises(WebSocketDisconnect) as disconnect:
            websocket.receive_text()

    assert disconnect.value.code == expected_code
    assert app.open_calls == []
    assert app.stop_calls == [("front", "session-1")]


def test_talk_websocket_rejects_start_message_without_type() -> None:
    """Start control messages must be explicit, not inferred from defaults."""
    # Given: A prepared talk WebSocket receives input fields without an explicit type.
    # When: The client sends the incomplete start control message.
    # Then: The API rejects it without inferring defaults or opening runtime talk.
    app = _StubTalkApp()
    client = _client(app)

    with client.websocket_connect(
        "/api/v1/talk/cameras/front/sessions/session-1/stream"
    ) as websocket:
        websocket.send_text(json.dumps({"codec": "pcm_s16le", "sample_rate": 16000}))
        with pytest.raises(WebSocketDisconnect) as disconnect:
            websocket.receive_text()

    assert disconnect.value.code == 1008
    assert app.open_calls == []
    assert app.stop_calls == [("front", "session-1")]


def test_talk_websocket_cleans_up_when_runtime_stream_open_fails() -> None:
    """Runtime restart/stale-socket failures during attach should stop the reservation."""
    # Given: Runtime fails while attaching the reserved talk WebSocket stream.
    # When: The client sends a valid start control message.
    # Then: The API closes the socket and stops the reserved session.
    app = _StubTalkApp(open_error=TalkRuntimeUnavailableError("stale runtime socket"))
    client = _client(app)

    with (
        client.websocket_connect(
            "/api/v1/talk/cameras/front/sessions/session-1/stream"
        ) as websocket,
        pytest.raises(WebSocketDisconnect),
    ):
        websocket.send_text(_start_message(TalkInputFormat()))
        websocket.receive_text()

    assert app.open_calls == []
    assert app.stop_calls == [("front", "session-1")]


def test_talk_websocket_does_not_stop_when_camera_disappears_during_open() -> None:
    """A camera-not-found attach failure means the runtime did not reserve this session."""
    # Given: Runtime reports camera-not-found before it owns the talk stream.
    # When: The WebSocket start message triggers stream attachment.
    # Then: The API closes with camera-not-found and avoids a redundant stop call.
    app = _StubTalkApp(open_error=TalkCameraNotFoundError("missing"))
    client = _client(app)

    with (
        client.websocket_connect(
            "/api/v1/talk/cameras/front/sessions/session-1/stream"
        ) as websocket,
        pytest.raises(WebSocketDisconnect) as disconnect,
    ):
        websocket.send_text(_start_message(TalkInputFormat()))
        websocket.receive_text()

    assert disconnect.value.code == 1008
    assert disconnect.value.reason == "Camera not found"
    assert app.open_calls == []
    assert app.stop_calls == []


def test_talk_websocket_maps_typed_stream_open_refusal_to_policy_close() -> None:
    """Attach-time typed refusals should survive to WebSocket close semantics."""
    # Given: Runtime refuses stream attachment with a typed policy-level reason.
    # When: The client starts the prepared WebSocket stream.
    # Then: The API maps the refusal to a policy close and releases the reservation.
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
        websocket.send_text(_start_message(TalkInputFormat()))
        websocket.receive_text()

    assert disconnect.value.code == 1008
    assert disconnect.value.reason == "Talk stream refused"
    assert app.open_calls == []
    assert app.stop_calls == [("front", "session-1")]


def test_talk_websocket_maps_backpressure_refusal_to_internal_error_close() -> None:
    """Infrastructure/open failures should use retryable internal-error close semantics."""
    # Given: Runtime refuses stream attachment because of talk backpressure.
    # When: The client starts the prepared WebSocket stream.
    # Then: The API maps it to a retryable internal-error close and stops the session.
    app = _StubTalkApp(
        open_error=TalkStreamOpenRefused(
            "Backpressure while opening stream",
            reason=TalkRefusalReason.BACKPRESSURE,
        )
    )
    client = _client(app)

    with (
        client.websocket_connect(
            "/api/v1/talk/cameras/front/sessions/session-1/stream"
        ) as websocket,
        pytest.raises(WebSocketDisconnect) as disconnect,
    ):
        websocket.send_text(_start_message(TalkInputFormat()))
        websocket.receive_text()

    assert disconnect.value.code == 1011
    assert disconnect.value.reason == "Talk stream unavailable"
    assert app.stop_calls == [("front", "session-1")]


def test_talk_websocket_rejects_non_binary_frame_after_ready() -> None:
    """Once active, only binary PCM frames or an exact stop control message are valid."""
    # Given: A talk WebSocket is active and ready for binary PCM frames.
    # When: The client sends an invalid text control frame after ready.
    # Then: The API closes the stream as a protocol error and stops runtime talk.
    app = _StubTalkApp()
    client = _client(app)

    with client.websocket_connect(
        "/api/v1/talk/cameras/front/sessions/session-1/stream"
    ) as websocket:
        websocket.send_text(_start_message(TalkInputFormat()))
        websocket.receive_json()
        websocket.send_text(json.dumps({"type": "stop", "extra": "not allowed"}))
        with pytest.raises(WebSocketDisconnect) as disconnect:
            websocket.receive_text()

    assert disconnect.value.code == 1003
    assert disconnect.value.reason == "Binary audio frames required"
    assert app.stop_calls == [("front", "session-1")]


def test_talk_websocket_closes_when_runtime_ipc_write_fails() -> None:
    """Runtime IPC write failures should close the browser stream without buffering audio."""
    # Given: A runtime talk writer fails while draining audio IPC.
    # When: The client sends a valid PCM frame over the WebSocket.
    # Then: The API closes with a retryable stream error and releases the session.
    input_format = TalkInputFormat(sample_rate=8000, frame_ms=10)
    writer = _MemoryTalkWriter(drain_error=BrokenPipeError("runtime gone"))
    app = _StubTalkApp(
        talk_config=TalkConfig(enabled=True, input=input_format),
        stream_writer=writer,
    )
    client = _client(app)

    with client.websocket_connect(
        "/api/v1/talk/cameras/front/sessions/session-1/stream"
    ) as websocket:
        websocket.send_text(_start_message(input_format))
        websocket.receive_json()
        websocket.send_bytes(b"\x44" * input_format.expected_bytes_per_frame)
        with pytest.raises(WebSocketDisconnect) as disconnect:
            websocket.receive_text()

    assert disconnect.value.code == 1011
    assert disconnect.value.reason == "Talk stream unavailable"
    assert writer.closed is True
    assert app.stop_calls == [("front", "session-1")]


def test_talk_websocket_ignores_disconnect_error_while_closing_runtime_stream() -> None:
    """Cleanup should tolerate a runtime writer that is already closed/reset."""
    # Given: Runtime writer cleanup raises because the stream is already reset.
    # When: The client stops an otherwise valid talk WebSocket session.
    # Then: The API treats cleanup as best-effort and records the stop call.
    input_format = TalkInputFormat(sample_rate=8000, frame_ms=10)
    writer = _MemoryTalkWriter(wait_closed_error=BrokenPipeError("already closed"))
    app = _StubTalkApp(
        talk_config=TalkConfig(enabled=True, input=input_format),
        stream_writer=writer,
    )
    client = _client(app)

    with client.websocket_connect(
        "/api/v1/talk/cameras/front/sessions/session-1/stream"
    ) as websocket:
        websocket.send_text(_start_message(input_format))
        websocket.receive_json()
        websocket.send_text(_stop_message())
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_text()

    assert writer.closed is True
    assert app.stop_calls == [("front", "session-1")]


def test_talk_websocket_accepts_short_lived_token_when_auth_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Browser WebSocket streams should authorize via the token returned by prepare."""
    # Given: Authenticated session preparation returns a short-lived talk stream token.
    # When: The browser connects with that token and sends one audio frame.
    # Then: The WebSocket authorizes and forwards the frame to runtime.
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    input_format = TalkInputFormat(sample_rate=8000, frame_ms=10)
    writer = _MemoryTalkWriter()
    app = _StubTalkApp(
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY"),
        talk_config=TalkConfig(enabled=True, input=input_format),
        prepare_result=CameraTalkSessionPrepared(
            camera_name="front",
            session_id="session-1",
            input=input_format,
        ),
        stream_writer=writer,
    )
    client = _client(app)
    prepare = client.post(
        "/api/v1/talk/cameras/front/sessions",
        headers={"Authorization": "Bearer secret"},
        json={"session_id": "session-1", "input": input_format.model_dump(mode="json")},
    )
    stream_url = prepare.json()["websocket_url"]
    frame = b"\x33" * input_format.expected_bytes_per_frame

    with client.websocket_connect(stream_url) as websocket:
        websocket.send_text(_start_message(input_format))
        ready = websocket.receive_json()
        websocket.send_bytes(frame)
        websocket.send_text(_stop_message())
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_text()

    assert ready["type"] == "ready"
    assert app.open_calls == [("front", "session-1", input_format)]
    assert bytes(writer.data[:4]) == len(frame).to_bytes(4, "big")
    assert bytes(writer.data[4:]) == frame


def test_talk_websocket_accepts_bearer_api_key_when_auth_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-browser clients may authorize the stream directly with the normal bearer key."""
    # Given: API authentication is enabled and a non-browser client has the bearer key.
    # When: The client opens the talk WebSocket with that bearer credential.
    # Then: The API authorizes the stream without a session token.
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    app = _StubTalkApp(
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY")
    )
    client = _client(app)

    with client.websocket_connect(
        "/api/v1/talk/cameras/front/sessions/session-1/stream",
        headers={"Authorization": "Bearer secret"},
    ) as websocket:
        websocket.send_text(_start_message(TalkInputFormat()))
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_text(_stop_message())
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_text()

    assert app.open_calls == [("front", "session-1", TalkInputFormat())]


def test_talk_websocket_rejects_wrong_bearer_api_key_without_runtime_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrong bearer credentials should fail auth without stopping a reserved session."""
    # Given: API authentication is enabled and the WebSocket bearer key is wrong.
    # When: The client attempts to open the talk stream.
    # Then: The API rejects auth before opening or stopping runtime talk.
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    app = _StubTalkApp(
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY")
    )
    client = _client(app)

    with (
        client.websocket_connect(
            "/api/v1/talk/cameras/front/sessions/session-1/stream",
            headers={"Authorization": "Bearer wrong-secret"},
        ) as websocket,
        pytest.raises(WebSocketDisconnect) as disconnect,
    ):
        websocket.receive_text()

    assert disconnect.value.code == 1008
    assert app.open_calls == []
    assert app.stop_calls == []


def test_talk_websocket_rejects_auth_when_api_key_missing() -> None:
    """Auth-enabled WebSockets should not fall back open if the API key is absent."""
    # Given: WebSocket token auth is enabled but the API key env var is missing.
    # When: The client connects with an arbitrary talk token.
    # Then: The API fails closed without opening or stopping runtime talk.
    app = _StubTalkApp(
        server_config=FastAPIServerConfig(
            auth_enabled=True,
            api_key_env="HOMESEC_MISSING_TEST_API_KEY",
        )
    )
    client = _client(app)

    with (
        client.websocket_connect(
            "/api/v1/talk/cameras/front/sessions/session-1/stream?token=anything"
        ) as websocket,
        pytest.raises(WebSocketDisconnect) as disconnect,
    ):
        websocket.receive_text()

    assert disconnect.value.code == 1011
    assert app.open_calls == []
    assert app.stop_calls == []


def test_talk_websocket_rejects_missing_token_when_auth_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authenticated WebSocket streams should fail closed without a bearer key or stream token."""
    # Given: API authentication is enabled for talk WebSocket streams.
    # When: The client connects without a bearer key or talk token.
    # Then: The API rejects the stream before runtime side effects.
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    app = _StubTalkApp(
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY")
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
    assert app.open_calls == []
    assert app.stop_calls == []


def test_talk_websocket_rejects_expired_token_when_auth_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expired talk tokens should be rejected before runtime stream attachment."""
    # Given: A signed talk stream token has already expired.
    # When: The client connects using that token.
    # Then: The API rejects the WebSocket before opening runtime talk.
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    token, _ = issue_camera_talk_token(
        api_key="secret",
        camera_name="front",
        session_id="session-1",
        ttl_s=1,
        now=datetime(2020, 1, 1, tzinfo=UTC),
    )
    app = _StubTalkApp(
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY")
    )
    client = _client(app)

    with (
        client.websocket_connect(
            f"/api/v1/talk/cameras/front/sessions/session-1/stream?token={token}"
        ) as websocket,
        pytest.raises(WebSocketDisconnect) as disconnect,
    ):
        websocket.receive_text()

    assert disconnect.value.code == 1008
    assert app.open_calls == []
    assert app.stop_calls == []


def test_talk_websocket_rejects_wrong_purpose_token_when_auth_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrong-purpose signed tokens should not authorize talk WebSocket streams."""
    # Given: A signed token is validly signed but scoped for preview streaming.
    # When: The client uses it for a talk WebSocket stream.
    # Then: The API rejects the wrong-purpose token without runtime side effects.
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    expires_at = int(datetime(2099, 1, 1, tzinfo=UTC).timestamp())
    payload_json = json.dumps(
        {
            "camera_name": "front",
            "exp": expires_at,
            "scope": "preview_stream",
            "session_id": "session-1",
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    payload_segment = talk_tokens._base64url_encode(payload_json)
    signature = talk_tokens._base64url_encode(
        talk_tokens._sign("secret", talk_tokens._signing_input(payload_segment))
    )
    token = f"{talk_tokens.TOKEN_VERSION}.{payload_segment}.{signature}"
    app = _StubTalkApp(
        server_config=FastAPIServerConfig(auth_enabled=True, api_key_env="HOMESEC_API_KEY")
    )
    client = _client(app)

    with (
        client.websocket_connect(
            f"/api/v1/talk/cameras/front/sessions/session-1/stream?token={token}"
        ) as websocket,
        pytest.raises(WebSocketDisconnect) as disconnect,
    ):
        websocket.receive_text()

    assert disconnect.value.code == 1008
    assert app.open_calls == []
    assert app.stop_calls == []
