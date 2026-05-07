"""Tests for RTSP talk backend registration behavior."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from homesec.models.config import CameraTalkConfig, TalkConfig
from homesec.models.talk import (
    TalkCapabilityProbeResult,
    TalkCapabilityState,
    TalkSessionOpenRequest,
)
from homesec.sources.rtsp.talk.backend import validate_rtsp_talk_backend_config
from homesec.talk.backends import (
    TalkBackendContext,
    TalkBackendRegistration,
    TalkBackendRegistry,
)


class _FutureBackendConfig(BaseModel):
    marker: str


class _FakeSession:
    session_id = "tk_fake"
    camera_name = "front"
    selected_codec = "PCMU/8000"

    async def write_pcm_frame(self, frame: bytes) -> None:
        _ = frame

    async def close(self) -> None:
        return None


class _FakeBackend:
    name = "future_backend"

    @property
    def supported_codecs(self) -> list[str]:
        return ["PCMU/8000"]

    async def probe(self) -> TalkCapabilityProbeResult:
        return TalkCapabilityProbeResult(capability=TalkCapabilityState.SUPPORTED)

    async def open_session(self, request: TalkSessionOpenRequest) -> _FakeSession:
        _ = request
        return _FakeSession()


def _future_registry() -> TalkBackendRegistry:
    registry = TalkBackendRegistry()
    registry.register(
        TalkBackendRegistration(
            name="future_backend",
            config_model=_FutureBackendConfig,
            factory=lambda config, context: _FakeBackend(),
        )
    )
    return registry


def _context(camera_talk: CameraTalkConfig) -> TalkBackendContext:
    return TalkBackendContext(
        camera_name="front",
        source_backend="rtsp",
        runtime_talk=TalkConfig(),
        camera_talk=camera_talk,
        resolved_source_uri="rtsp://camera.local/stream1",
    )


def test_rtsp_talk_config_validation_uses_registered_explicit_backend_model() -> None:
    """Explicit registered backends should validate through the registry contract."""
    # Given: A registered future backend with required backend-specific config
    context = _context(
        CameraTalkConfig(
            backend="future_backend",
            backends={"future_backend": {"marker": "configured"}},
        )
    )

    # When: Validating RTSP talk backend config at source-config load time
    validate_rtsp_talk_backend_config(context, registry=_future_registry())

    # Then: The registered backend config is accepted without ONVIF-specific branches


def test_rtsp_talk_config_validation_rejects_registered_backend_config_errors() -> None:
    """Explicit registered backend config errors should fail source-config validation."""
    # Given: A registered future backend with invalid backend-specific config
    context = _context(
        CameraTalkConfig(
            backend="future_backend",
            backends={"future_backend": {}},
        )
    )

    # When: Validating RTSP talk backend config at source-config load time
    # Then: The selected backend's Pydantic model reports the config error
    with pytest.raises(ValidationError, match="marker"):
        validate_rtsp_talk_backend_config(context, registry=_future_registry())
