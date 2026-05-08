"""RTSP talk backend registrations."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel

from homesec.models.talk import (
    TalkCapabilityProbeResult,
    TalkRefusalReason,
    TalkSessionOpenRequest,
)
from homesec.sources.rtsp.talk.errors import TalkProtocolError, TalkProtocolErrorCode
from homesec.sources.rtsp.talk.models import ONVIFBackchannelConfig
from homesec.sources.rtsp.talk.onvif_backchannel import ONVIFBackchannelAdapter
from homesec.talk.backends import (
    TalkBackendAdapter,
    TalkBackendConfigError,
    TalkBackendContext,
    TalkBackendOpenError,
    TalkBackendRegistration,
    TalkBackendRegistry,
    TalkBackendSession,
    backend_config_for,
    model_validate_backend_config,
)

ONVIF_RTSP_BACKCHANNEL_BACKEND = "onvif_rtsp_backchannel"
_SAFE_ENV_VAR_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_MISSING_ENV_PREFIXES = (
    "RTSP URL environment variable is not set: ",
    "RTSP credential environment variable is not set: ",
)


@dataclass(slots=True)
class ONVIFRTSPTalkBackendAdapter:
    """Talk backend wrapper around the existing ONVIF RTSP backchannel adapter."""

    config: ONVIFBackchannelConfig
    adapter: ONVIFBackchannelAdapter
    name: str = ONVIF_RTSP_BACKCHANNEL_BACKEND

    @property
    def supported_codecs(self) -> list[str]:
        """Return codecs supported by this ONVIF backend implementation."""
        return list(self.config.preferred_codecs)

    async def probe(self) -> TalkCapabilityProbeResult:
        """Probe camera ONVIF RTSP backchannel capability."""
        return await self.adapter.probe()

    async def open_session(self, request: TalkSessionOpenRequest) -> TalkBackendSession:
        """Open an ONVIF RTSP backchannel session."""
        try:
            return await self.adapter.open_session(
                session_id=request.session_id,
                input_sample_rate=request.input.sample_rate,
            )
        except TalkProtocolError as exc:
            raise _backend_open_error_from_protocol_error(exc) from exc
        except (asyncio.TimeoutError, TimeoutError, EOFError, OSError) as exc:
            raise TalkBackendOpenError(
                "Camera talk backchannel failed",
                reason=TalkRefusalReason.CAMERA_BACKCHANNEL_FAILED,
            ) from exc


def build_rtsp_talk_backend_registry() -> TalkBackendRegistry:
    """Build the built-in talk backend registry.

    Deprecated compatibility alias. Use ``build_default_talk_backend_registry`` for
    new source wiring so future non-RTSP talk backends do not inherit RTSP-only
    naming.
    """
    from homesec.talk.registry import build_default_talk_backend_registry

    return build_default_talk_backend_registry()


def onvif_rtsp_talk_backend_registration() -> TalkBackendRegistration:
    """Return the built-in ONVIF RTSP talk backend registration."""
    return TalkBackendRegistration(
        name=ONVIF_RTSP_BACKCHANNEL_BACKEND,
        config_model=ONVIFBackchannelConfig,
        config_factory=_validate_onvif_config,
        factory=_build_onvif_backend,
        priority=10,
        standards_based=True,
    )


def validate_rtsp_talk_backend_config(
    context: TalkBackendContext,
    *,
    registry: TalkBackendRegistry | None = None,
) -> None:
    """Validate statically known RTSP talk backend config at source-config load time."""
    registry = registry or build_rtsp_talk_backend_registry()
    camera_talk = context.camera_talk
    if camera_talk.backend == "auto":
        for registration in registry.standards_first():
            if registration.standards_based:
                _validate_registered_backend_config(registration, context)
        return

    explicit_registration = registry.get(camera_talk.backend)
    if explicit_registration is None:
        return
    _validate_registered_backend_config(explicit_registration, context)


def _validate_registered_backend_config(
    registration: TalkBackendRegistration,
    context: TalkBackendContext,
) -> None:
    model_validate_backend_config(
        registration,
        backend_config_for(context.camera_talk, registration.name),
        context=context,
    )


def _build_onvif_backchannel_config_data(
    value: Mapping[str, object] | BaseModel,
    *,
    fallback_rtsp_url: str | None,
    fallback_rtsp_url_env: str | None = None,
) -> dict[str, object]:
    """Build ONVIF talk adapter config with inherited source URL fallback rules."""
    adapter_config_data = _talk_config_dict(value)
    if (
        adapter_config_data.get("rtsp_url") is not None
        or adapter_config_data.get("rtsp_url_env") is not None
    ):
        return adapter_config_data
    if fallback_rtsp_url is not None:
        adapter_config_data["rtsp_url"] = fallback_rtsp_url
    elif fallback_rtsp_url_env is not None:
        adapter_config_data["rtsp_url_env"] = fallback_rtsp_url_env
    return adapter_config_data


def _talk_config_dict(value: Mapping[str, object] | BaseModel) -> dict[str, object]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value)


def _validate_onvif_config(
    raw_config: Mapping[str, object] | BaseModel,
    context: TalkBackendContext,
) -> ONVIFBackchannelConfig:
    return ONVIFBackchannelConfig.model_validate(
        _build_onvif_backchannel_config_data(
            raw_config,
            fallback_rtsp_url=context.resolved_source_uri or context.source_uri,
            fallback_rtsp_url_env=context.source_uri_env,
        )
    )


def _build_onvif_backend(
    config: BaseModel,
    context: TalkBackendContext,
) -> TalkBackendAdapter:
    if not isinstance(config, ONVIFBackchannelConfig):
        raise TypeError(f"Expected ONVIFBackchannelConfig, got {type(config).__name__}")
    try:
        rtsp_url = config.resolve_rtsp_url()
        adapter = ONVIFBackchannelAdapter(
            config,
            rtsp_url=rtsp_url,
            camera_name=context.camera_name,
        )
    except ValueError as exc:
        raise TalkBackendConfigError(_safe_onvif_config_error(exc), cause=exc) from exc
    return ONVIFRTSPTalkBackendAdapter(config=config, adapter=adapter)


def _safe_onvif_config_error(exc: ValueError) -> str:
    base = f"Talk backend '{ONVIF_RTSP_BACKCHANNEL_BACKEND}' config is invalid"
    detail = str(exc).strip()
    if not detail:
        return base
    for prefix in _MISSING_ENV_PREFIXES:
        if detail.startswith(prefix):
            env_name = detail.removeprefix(prefix).strip()
            if _SAFE_ENV_VAR_NAME_PATTERN.fullmatch(env_name):
                return f"{base}: {prefix}{env_name}"
            return base
    return base


def _backend_open_error_from_protocol_error(exc: TalkProtocolError) -> TalkBackendOpenError:
    match exc.code:
        case TalkProtocolErrorCode.CAMERA_BACKCHANNEL_UNSUPPORTED:
            return TalkBackendOpenError(
                "Camera does not advertise an ONVIF talk backchannel",
                reason=TalkRefusalReason.UNSUPPORTED_CAMERA,
            )
        case TalkProtocolErrorCode.UNSUPPORTED_CODEC:
            return TalkBackendOpenError(
                "Camera talk backchannel codec is not supported",
                reason=TalkRefusalReason.UNSUPPORTED_CODEC,
            )
        case TalkProtocolErrorCode.RTSP_AUTH_FAILED:
            return TalkBackendOpenError(
                "Camera talk backchannel authentication failed",
                reason=TalkRefusalReason.TALK_AUTH_FAILED,
            )
        case (
            TalkProtocolErrorCode.CAMERA_REJECTED_SESSION
            | TalkProtocolErrorCode.CAMERA_STREAM_FAILED
            | TalkProtocolErrorCode.RTSP_PROTOCOL_ERROR
        ):
            return TalkBackendOpenError(
                "Camera talk backchannel failed",
                reason=TalkRefusalReason.CAMERA_BACKCHANNEL_FAILED,
            )
