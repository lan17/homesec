"""Push-to-talk control-plane and WebSocket streaming endpoints."""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast
from urllib.parse import quote, urlencode

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field, ValidationError

from homesec.api.dependencies import get_homesec_app
from homesec.api.errors import APIError, APIErrorCode
from homesec.api.talk_tokens import (
    TalkTokenError,
    issue_camera_talk_token,
    validate_camera_talk_token,
)
from homesec.models.talk import (
    CameraTalkStatus,
    TalkCapabilityState,
    TalkInputFormat,
    TalkRefusalReason,
    TalkState,
)
from homesec.runtime.errors import (
    TalkCameraNotFoundError,
    TalkRuntimeUnavailableError,
    TalkStreamOpenRefused,
)
from homesec.runtime.ipc_stream import IPCFrameError, write_length_prefixed_frame
from homesec.runtime.models import (
    CameraTalkStartRefusal,
    CameraTalkStopResult,
    RuntimeTalkStream,
)

if TYPE_CHECKING:
    from homesec.app import Application

logger = logging.getLogger(__name__)
_SESSION_ID_PATTERN = r"^[A-Za-z0-9._:-]+$"
_TALK_WS_START_TYPE: Literal["start"] = "start"
_TALK_WS_READY_TYPE = "ready"

control_router = APIRouter(tags=["talk"])
stream_router = APIRouter(tags=["talk"])
router = control_router


class TalkStatusResponse(BaseModel):
    camera_name: str
    enabled: bool
    policy_enabled: bool
    capability: TalkCapabilityState
    state: TalkState
    active_session_id: str | None = None
    supported_codecs: list[str] = Field(default_factory=list)
    offered_codecs: list[str] = Field(default_factory=list)
    selected_codec: str | None = None
    last_error: str | None = None


class TalkSessionRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=_SESSION_ID_PATTERN,
    )
    input: TalkInputFormat | None = None


class TalkSessionResponse(BaseModel):
    camera_name: str
    session_id: str
    state: TalkState
    input: TalkInputFormat
    websocket_url: str
    stream_url: str
    token: str | None = None
    token_expires_at: datetime | None = None
    max_session_s: int
    idle_timeout_s: float


class TalkWebSocketStartMessage(BaseModel):
    """Client control message that starts a prepared talk stream."""

    model_config = {"extra": "forbid"}

    type: Literal["start"]
    codec: Literal["pcm_s16le"]
    sample_rate: int = Field(ge=8000, le=48000)
    channels: int = Field(ge=1, le=1)
    frame_ms: int = Field(ge=10, le=60)

    def input_format(self) -> TalkInputFormat:
        return TalkInputFormat(
            codec=self.codec,
            sample_rate=self.sample_rate,
            channels=self.channels,
            frame_ms=self.frame_ms,
        )


class TalkWebSocketStopMessage(BaseModel):
    """Client control message that ends a talk stream."""

    model_config = {"extra": "forbid"}

    type: Literal["stop"]


class TalkStopResponse(BaseModel):
    accepted: bool
    state: TalkState


def _status_response(talk_status: CameraTalkStatus) -> TalkStatusResponse:
    return TalkStatusResponse(
        camera_name=talk_status.camera_name,
        enabled=talk_status.enabled,
        policy_enabled=talk_status.policy_enabled,
        capability=talk_status.capability,
        state=talk_status.state,
        active_session_id=talk_status.active_session_id,
        supported_codecs=list(talk_status.supported_codecs),
        offered_codecs=list(talk_status.offered_codecs),
        selected_codec=talk_status.selected_codec,
        last_error=talk_status.last_error,
    )


def _stream_url(
    camera_name: str,
    session_id: str,
    *,
    token: str | None = None,
) -> str:
    path = (
        f"/api/v1/talk/cameras/{quote(camera_name, safe='')}"
        f"/sessions/{quote(session_id, safe='')}/stream"
    )
    if token is None:
        return path
    return f"{path}?{urlencode({'token': token})}"


def _new_session_id() -> str:
    return "tk_" + secrets.token_urlsafe(16)


def _refusal_status_and_code(reason: TalkRefusalReason) -> tuple[int, APIErrorCode]:
    match reason:
        case TalkRefusalReason.CAMERA_NOT_FOUND:
            return status.HTTP_404_NOT_FOUND, APIErrorCode.TALK_CAMERA_NOT_FOUND
        case TalkRefusalReason.TALK_DISABLED:
            return status.HTTP_409_CONFLICT, APIErrorCode.TALK_DISABLED
        case TalkRefusalReason.SOURCE_NOT_TALK_CAPABLE:
            return status.HTTP_409_CONFLICT, APIErrorCode.TALK_SOURCE_NOT_TALK_CAPABLE
        case TalkRefusalReason.SESSION_ALREADY_ACTIVE:
            return status.HTTP_409_CONFLICT, APIErrorCode.TALK_SESSION_ALREADY_ACTIVE
        case TalkRefusalReason.SESSION_BUDGET_EXHAUSTED:
            return status.HTTP_409_CONFLICT, APIErrorCode.TALK_SESSION_BUDGET_EXHAUSTED
        case TalkRefusalReason.UNSUPPORTED_CAMERA:
            return status.HTTP_409_CONFLICT, APIErrorCode.TALK_UNSUPPORTED_CAMERA
        case TalkRefusalReason.UNSUPPORTED_CODEC:
            return status.HTTP_400_BAD_REQUEST, APIErrorCode.TALK_UNSUPPORTED_CODEC
        case TalkRefusalReason.CAMERA_BACKCHANNEL_FAILED:
            return status.HTTP_503_SERVICE_UNAVAILABLE, APIErrorCode.TALK_CAMERA_BACKCHANNEL_FAILED
        case TalkRefusalReason.RUNTIME_UNAVAILABLE:
            return status.HTTP_503_SERVICE_UNAVAILABLE, APIErrorCode.TALK_RUNTIME_UNAVAILABLE
        case TalkRefusalReason.INVALID_AUDIO_FRAME:
            return status.HTTP_400_BAD_REQUEST, APIErrorCode.TALK_INVALID_AUDIO_FRAME
        case TalkRefusalReason.BACKPRESSURE:
            return status.HTTP_503_SERVICE_UNAVAILABLE, APIErrorCode.TALK_BACKPRESSURE


def _raise_talk_refusal(refusal: CameraTalkStartRefusal) -> NoReturn:
    status_code, error_code = _refusal_status_and_code(refusal.reason)
    raise APIError(
        refusal.message,
        status_code=status_code,
        error_code=error_code,
        extra={"reason": refusal.reason.value},
    )


def _raise_camera_not_found(exc: TalkCameraNotFoundError) -> NoReturn:
    raise APIError(
        str(exc),
        status_code=status.HTTP_404_NOT_FOUND,
        error_code=APIErrorCode.TALK_CAMERA_NOT_FOUND,
    ) from exc


def _raise_runtime_unavailable(exc: TalkRuntimeUnavailableError) -> NoReturn:
    raise APIError(
        str(exc),
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        error_code=APIErrorCode.TALK_RUNTIME_UNAVAILABLE,
    ) from exc


@control_router.get("/api/v1/talk/cameras/{camera_name}", response_model=TalkStatusResponse)
async def get_talk_status(
    camera_name: str,
    app: Application = Depends(get_homesec_app),
) -> TalkStatusResponse:
    """Return push-to-talk status for a camera."""
    try:
        talk_status = await app.get_camera_talk_status(camera_name)
    except TalkCameraNotFoundError as exc:
        _raise_camera_not_found(exc)
    except TalkRuntimeUnavailableError as exc:
        _raise_runtime_unavailable(exc)

    return _status_response(talk_status)


@control_router.post(
    "/api/v1/talk/cameras/{camera_name}/sessions",
    response_model=TalkSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def prepare_talk_session(
    camera_name: str,
    request: TalkSessionRequest | None = None,
    app: Application = Depends(get_homesec_app),
) -> TalkSessionResponse:
    """Reserve a push-to-talk session slot and return the WebSocket stream URL."""
    body = request or TalkSessionRequest()
    requested_session_id = body.session_id or _new_session_id()
    input_format = body.input or app.config.talk.input
    try:
        outcome = await app.prepare_camera_talk_session(
            camera_name,
            session_id=requested_session_id,
            input_format=input_format,
        )
    except TalkCameraNotFoundError as exc:
        _raise_camera_not_found(exc)
    except TalkRuntimeUnavailableError as exc:
        _raise_runtime_unavailable(exc)

    if isinstance(outcome, CameraTalkStartRefusal):
        _raise_talk_refusal(outcome)

    talk_config = app.config.talk
    token: str | None = None
    expires_at: datetime | None = None
    if app.server_config.auth_enabled:
        api_key = app.server_config.get_api_key()
        if api_key is None:
            raise APIError(
                "API key not configured",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error_code=APIErrorCode.API_KEY_NOT_CONFIGURED,
            )
        token, expires_at = issue_camera_talk_token(
            api_key=api_key,
            camera_name=camera_name,
            session_id=outcome.session_id,
            ttl_s=talk_config.token_ttl_s,
        )

    websocket_url = _stream_url(
        camera_name,
        outcome.session_id,
        token=token,
    )
    return TalkSessionResponse(
        camera_name=outcome.camera_name,
        session_id=outcome.session_id,
        state=TalkState.STARTING,
        input=outcome.input,
        websocket_url=websocket_url,
        stream_url=websocket_url,
        token=token,
        token_expires_at=expires_at,
        max_session_s=talk_config.max_session_s,
        idle_timeout_s=talk_config.idle_timeout_s,
    )


@control_router.delete(
    "/api/v1/talk/cameras/{camera_name}/sessions/{session_id}",
    response_model=TalkStopResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def stop_talk_session(
    camera_name: str,
    session_id: str,
    app: Application = Depends(get_homesec_app),
) -> TalkStopResponse:
    """Stop a reserved or active push-to-talk session."""
    try:
        result = await app.stop_camera_talk_session(camera_name, session_id=session_id)
    except TalkCameraNotFoundError as exc:
        _raise_camera_not_found(exc)
    except TalkRuntimeUnavailableError as exc:
        _raise_runtime_unavailable(exc)

    return _stop_response(result)


@stream_router.websocket("/api/v1/talk/cameras/{camera_name}/sessions/{session_id}/stream")
async def stream_talk_audio(
    websocket: WebSocket,
    camera_name: str,
    session_id: str,
) -> None:
    """Bridge browser PCM frames into the runtime worker talk stream."""
    app = await _get_websocket_app(websocket)
    if app is None:
        return

    await websocket.accept()
    if not await _authorize_talk_websocket(websocket, app, camera_name, session_id):
        return

    input_format = await _receive_talk_start_message(websocket)
    if input_format is None:
        await _stop_talk_session_best_effort(app, camera_name, session_id)
        return

    stream: RuntimeTalkStream | None = None
    stop_best_effort = True
    try:
        try:
            stream = await app.open_camera_talk_stream(
                camera_name,
                session_id=session_id,
                input_format=input_format,
            )
        except TalkCameraNotFoundError:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Camera not found")
            stop_best_effort = False
            return
        except TalkStreamOpenRefused as exc:
            logger.info(
                "Talk WebSocket open refused for camera=%s session=%s reason=%s: %s",
                camera_name,
                session_id,
                exc.reason.value,
                exc,
            )
            await _stop_talk_session_best_effort(app, camera_name, session_id)
            stop_best_effort = False
            close_code, close_reason = _talk_stream_refusal_close(exc.reason)
            await websocket.close(code=close_code, reason=close_reason)
            return
        except TalkRuntimeUnavailableError as exc:
            logger.info(
                "Talk WebSocket open refused for camera=%s session=%s: %s",
                camera_name,
                session_id,
                exc,
            )
            await _stop_talk_session_best_effort(app, camera_name, session_id)
            stop_best_effort = False
            await websocket.close(
                code=status.WS_1011_INTERNAL_ERROR,
                reason="Talk stream unavailable",
            )
            return

        await websocket.send_json(
            {
                "type": _TALK_WS_READY_TYPE,
                "camera_name": camera_name,
                "session_id": session_id,
                "input": input_format.model_dump(mode="json"),
                "camera_codec": stream.selected_codec,
            }
        )
        await _forward_websocket_frames(websocket, stream, input_format)
    finally:
        if stream is not None:
            await _close_talk_stream_writer(stream)
        if stop_best_effort:
            await _stop_talk_session_best_effort(app, camera_name, session_id)


def _stop_response(result: CameraTalkStopResult) -> TalkStopResponse:
    return TalkStopResponse(accepted=result.accepted, state=result.state)


def _talk_stream_refusal_close(reason: TalkRefusalReason) -> tuple[int, str]:
    """Map typed stream-open refusals to WebSocket close semantics."""
    match reason:
        case (
            TalkRefusalReason.CAMERA_NOT_FOUND
            | TalkRefusalReason.TALK_DISABLED
            | TalkRefusalReason.SOURCE_NOT_TALK_CAPABLE
            | TalkRefusalReason.SESSION_ALREADY_ACTIVE
            | TalkRefusalReason.SESSION_BUDGET_EXHAUSTED
            | TalkRefusalReason.UNSUPPORTED_CAMERA
            | TalkRefusalReason.UNSUPPORTED_CODEC
            | TalkRefusalReason.INVALID_AUDIO_FRAME
        ):
            return status.WS_1008_POLICY_VIOLATION, "Talk stream refused"
        case (
            TalkRefusalReason.CAMERA_BACKCHANNEL_FAILED
            | TalkRefusalReason.RUNTIME_UNAVAILABLE
            | TalkRefusalReason.BACKPRESSURE
        ):
            return status.WS_1011_INTERNAL_ERROR, "Talk stream unavailable"


async def _get_websocket_app(websocket: WebSocket) -> Application | None:
    app = cast("Application | None", getattr(websocket.app.state, "homesec", None))
    if app is None:
        await websocket.close(
            code=status.WS_1011_INTERNAL_ERROR, reason="Application not initialized"
        )
        return None
    if app.bootstrap_mode:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Setup required")
        return None
    return app


async def _authorize_talk_websocket(
    websocket: WebSocket,
    app: Application,
    camera_name: str,
    session_id: str,
) -> bool:
    server_config = app.server_config
    if not server_config.auth_enabled:
        return True

    api_key = server_config.get_api_key()
    if not api_key:
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="API key not configured")
        return False

    bearer_token = _parse_websocket_bearer_token(websocket)
    if bearer_token is not None and secrets.compare_digest(bearer_token, api_key):
        return True

    token = websocket.query_params.get("token")
    if token is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Talk token rejected")
        return False

    try:
        validate_camera_talk_token(
            api_key=api_key,
            token=token,
            camera_name=camera_name,
            session_id=session_id,
        )
    except TalkTokenError as exc:
        logger.info(
            "Rejected talk token for camera_name=%s session_id=%s reason=%s",
            camera_name,
            session_id,
            exc.code.value,
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Talk token rejected")
        return False

    return True


def _parse_websocket_bearer_token(websocket: WebSocket) -> str | None:
    auth_header = websocket.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    return auth_header.removeprefix("Bearer ").strip()


async def _receive_talk_start_message(websocket: WebSocket) -> TalkInputFormat | None:
    try:
        message = await websocket.receive()
    except WebSocketDisconnect:
        return None
    if message.get("type") == "websocket.disconnect":
        return None

    text_payload = message.get("text")
    if text_payload is None:
        await websocket.close(
            code=status.WS_1003_UNSUPPORTED_DATA,
            reason="Talk start message required",
        )
        return None

    payload = _decode_talk_control_payload(text_payload)
    if payload is None:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Invalid talk start message",
        )
        return None

    try:
        return TalkWebSocketStartMessage.model_validate(payload).input_format()
    except ValidationError:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Invalid talk start message",
        )
        return None


def _decode_talk_control_payload(text_payload: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text_payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _is_talk_stop_message(text_payload: str) -> bool:
    payload = _decode_talk_control_payload(text_payload)
    if payload is None:
        return False
    try:
        TalkWebSocketStopMessage.model_validate(payload)
    except ValidationError:
        return False
    return True


async def _forward_websocket_frames(
    websocket: WebSocket,
    stream: RuntimeTalkStream,
    input_format: TalkInputFormat,
) -> None:
    expected_bytes = input_format.expected_bytes_per_frame
    while True:
        try:
            message = await websocket.receive()
        except WebSocketDisconnect:
            return
        message_type = message.get("type")
        if message_type == "websocket.disconnect":
            return

        payload = message.get("bytes")
        if payload is not None:
            if len(payload) != expected_bytes:
                await websocket.close(
                    code=status.WS_1003_UNSUPPORTED_DATA,
                    reason="Invalid audio frame length",
                )
                return
            try:
                await write_length_prefixed_frame(stream.writer, payload)
            except (BrokenPipeError, ConnectionResetError, IPCFrameError):
                await websocket.close(
                    code=status.WS_1011_INTERNAL_ERROR,
                    reason="Talk stream unavailable",
                )
                return
            continue

        text_payload = message.get("text")
        if text_payload is not None and _is_talk_stop_message(text_payload):
            await websocket.close(code=status.WS_1000_NORMAL_CLOSURE, reason="Talk stopped")
            return
        await websocket.close(
            code=status.WS_1003_UNSUPPORTED_DATA,
            reason="Binary audio frames required",
        )
        return


async def _close_talk_stream_writer(stream: RuntimeTalkStream) -> None:
    stream.writer.close()
    try:
        await stream.writer.wait_closed()
    except (BrokenPipeError, ConnectionResetError):
        pass


async def _stop_talk_session_best_effort(
    app: Application,
    camera_name: str,
    session_id: str,
) -> None:
    try:
        await app.stop_camera_talk_session(camera_name, session_id=session_id)
    except Exception as exc:  # pragma: no cover - defensive cleanup only
        logger.debug(
            "Talk session cleanup skipped for camera=%s session=%s: %s",
            camera_name,
            session_id,
            exc,
        )
