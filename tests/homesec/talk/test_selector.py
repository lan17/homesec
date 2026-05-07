"""Tests for talk backend selection behavior."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from homesec.models.config import CameraTalkConfig, TalkConfig
from homesec.models.talk import (
    TalkCapabilityProbeResult,
    TalkCapabilityState,
    TalkInputFormat,
    TalkRefusalReason,
    TalkSessionOpenRequest,
)
from homesec.talk.backends import (
    TalkBackendContext,
    TalkBackendOpenError,
    TalkBackendRegistration,
    TalkBackendRegistry,
)
from homesec.talk.selector import TalkBackendSelector


class _BackendConfig(BaseModel):
    marker: str = "default"


class _FakeSession:
    session_id = "tk_fake"
    camera_name = "cam"
    selected_codec = "PCMU/8000"

    async def write_pcm_frame(self, frame: bytes) -> None:
        _ = frame

    async def close(self) -> None:
        return None


class _FakeBackend:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self._calls = calls

    @property
    def supported_codecs(self) -> list[str]:
        return ["PCMU/8000"]

    async def probe(self) -> TalkCapabilityProbeResult:
        self._calls.append(f"{self.name}:probe")
        return TalkCapabilityProbeResult(capability=TalkCapabilityState.SUPPORTED)

    async def open_session(self, request: TalkSessionOpenRequest) -> _FakeSession:
        self._calls.append(f"{self.name}:open:{request.session_id}")
        return _FakeSession()


def _context(camera_talk: CameraTalkConfig) -> TalkBackendContext:
    return TalkBackendContext(
        camera_name="cam",
        source_backend="rtsp",
        runtime_talk=TalkConfig(),
        camera_talk=camera_talk,
    )


def _registry(calls: list[str]) -> TalkBackendRegistry:
    registry = TalkBackendRegistry()
    registry.register(
        TalkBackendRegistration(
            name="standard_backend",
            config_model=_BackendConfig,
            factory=lambda config, context: _FakeBackend("standard_backend", calls),
            priority=10,
            standards_based=True,
        )
    )
    registry.register(
        TalkBackendRegistration(
            name="vendor_backend",
            config_model=_BackendConfig,
            factory=lambda config, context: _FakeBackend("vendor_backend", calls),
            priority=1,
            standards_based=False,
        )
    )
    return registry


@pytest.mark.asyncio
async def test_selector_auto_uses_standards_backend_first() -> None:
    """Backend auto mode should prefer standards-based registrations."""
    # Given: A selector in auto mode with standards and vendor backend registrations
    calls: list[str] = []
    selector = TalkBackendSelector(
        registry=_registry(calls),
        context=_context(CameraTalkConfig(backend="auto")),
    )

    # When: Probing and opening a talk session
    probe = await selector.probe()
    session = await selector.open_session(
        TalkSessionOpenRequest(session_id="tk_auto", input=TalkInputFormat())
    )

    # Then: The standards backend handles the request before vendor-specific candidates
    assert probe.capability == TalkCapabilityState.SUPPORTED
    assert selector.supported_codecs == ["PCMU/8000"]
    assert session.selected_codec == "PCMU/8000"
    assert calls == ["standard_backend:probe", "standard_backend:open:tk_auto"]


@pytest.mark.asyncio
async def test_selector_explicit_backend_does_not_fallback_to_standards_backend() -> None:
    """Explicit backend mode should not fallback to registered standards backends."""
    # Given: A selector with an explicit backend that is not registered
    calls: list[str] = []
    selector = TalkBackendSelector(
        registry=_registry(calls),
        context=_context(CameraTalkConfig(backend="missing_vendor")),
    )

    # When: Probing and opening through the selector
    probe = await selector.probe()

    # Then: Selection reports a config/runtime error without probing fallback backends
    assert probe.capability == TalkCapabilityState.ERROR
    assert probe.refusal_reason == TalkRefusalReason.RUNTIME_UNAVAILABLE
    assert probe.message == "Talk backend 'missing_vendor' is not registered in this runtime"
    assert selector.supported_codecs == []
    assert calls == []

    # When: Opening a session with the same missing explicit backend
    # Then: The selector raises the same refusal reason for TalkManager mapping
    with pytest.raises(TalkBackendOpenError) as exc_info:
        await selector.open_session(
            TalkSessionOpenRequest(session_id="tk_missing", input=TalkInputFormat())
        )
    assert exc_info.value.reason == TalkRefusalReason.RUNTIME_UNAVAILABLE


@pytest.mark.asyncio
async def test_selector_uses_explicit_registered_backend_config() -> None:
    """Explicit registered backends should receive their own config block."""
    # Given: A selector with backend-specific config for an explicit vendor backend
    seen_config: list[_BackendConfig] = []
    registry = TalkBackendRegistry()
    registry.register(
        TalkBackendRegistration(
            name="vendor_backend",
            config_model=_BackendConfig,
            factory=lambda config, context: (
                seen_config.append(config) or _FakeBackend("vendor_backend", [])
            ),
        )
    )
    selector = TalkBackendSelector(
        registry=registry,
        context=_context(
            CameraTalkConfig(
                backend="vendor_backend",
                config={"marker": "legacy"},
                backends={"vendor_backend": {"marker": "backend-map"}},
            )
        ),
    )

    # When: Probing the explicitly selected backend
    probe = await selector.probe()

    # Then: Backend-specific config wins over the legacy config alias
    assert probe.capability == TalkCapabilityState.SUPPORTED
    assert seen_config == [_BackendConfig(marker="backend-map")]
