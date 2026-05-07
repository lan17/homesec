"""Readiness tests for non-ONVIF talk backend adapters."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel

from homesec.models.config import CameraTalkConfig, TalkConfig
from homesec.models.talk import (
    TalkCapabilityProbeResult,
    TalkCapabilityState,
    TalkInputFormat,
    TalkRefusalReason,
    TalkSessionOpenRequest,
    TalkSessionPrepareRequest,
)
from homesec.sources.rtsp.talk.manager import TalkManager
from homesec.talk.backends import (
    CameraTalkFingerprint,
    TalkBackendContext,
    TalkBackendDetection,
    TalkBackendRegistration,
    TalkBackendRegistry,
)
from homesec.talk.selector import TalkBackendSelector


class _FakeConfig(BaseModel):
    supported: bool = True


@dataclass(slots=True)
class _FakeProprietaryTalkSession:
    session_id: str
    camera_name: str = "front"
    selected_codec: str = "PCMA/8000"
    frames: list[bytes] = field(default_factory=list)
    closed: bool = False

    async def write_pcm_frame(self, frame: bytes) -> None:
        self.frames.append(frame)

    async def close(self) -> None:
        self.closed = True


class _StandardTalkBackend:
    name = "onvif_rtsp_backchannel"

    def __init__(
        self,
        *,
        calls: list[str],
        capability: TalkCapabilityState = TalkCapabilityState.UNSUPPORTED,
    ) -> None:
        self._calls = calls
        self._capability = capability

    @property
    def supported_codecs(self) -> list[str]:
        return ["PCMU/8000"]

    async def probe(self) -> TalkCapabilityProbeResult:
        self._calls.append("onvif:probe")
        if self._capability == TalkCapabilityState.SUPPORTED:
            return TalkCapabilityProbeResult(
                capability=TalkCapabilityState.SUPPORTED,
                offered_codecs=["PCMU/8000"],
                selected_codec="PCMU/8000",
            )
        return TalkCapabilityProbeResult(
            capability=TalkCapabilityState.UNSUPPORTED,
            refusal_reason=TalkRefusalReason.UNSUPPORTED_CAMERA,
            message="ONVIF backchannel unsupported",
        )

    async def open_session(
        self,
        request: TalkSessionOpenRequest,
    ) -> _FakeProprietaryTalkSession:
        self._calls.append(f"onvif:open:{request.session_id}")
        return _FakeProprietaryTalkSession(
            session_id=request.session_id,
            selected_codec="PCMU/8000",
        )


class _FakeProprietaryTalkBackend:
    name = "fake_proprietary"

    def __init__(
        self,
        *,
        config: _FakeConfig,
        calls: list[str],
        sessions: list[_FakeProprietaryTalkSession],
    ) -> None:
        self._config = config
        self._calls = calls
        self._sessions = sessions

    @property
    def supported_codecs(self) -> list[str]:
        return ["PCMA/8000"]

    async def probe(self) -> TalkCapabilityProbeResult:
        self._calls.append("fake:probe")
        if self._config.supported:
            return TalkCapabilityProbeResult(
                capability=TalkCapabilityState.SUPPORTED,
                offered_codecs=["PCMA/8000"],
                selected_codec="PCMA/8000",
            )
        return TalkCapabilityProbeResult(
            capability=TalkCapabilityState.UNSUPPORTED,
            refusal_reason=TalkRefusalReason.UNSUPPORTED_CAMERA,
            message="fake proprietary backend unsupported",
        )

    async def open_session(
        self,
        request: TalkSessionOpenRequest,
    ) -> _FakeProprietaryTalkSession:
        self._calls.append(f"fake:open:{request.session_id}")
        session = _FakeProprietaryTalkSession(session_id=request.session_id)
        self._sessions.append(session)
        return session


def _context(
    camera_talk: CameraTalkConfig,
    *,
    fingerprint: CameraTalkFingerprint | None = None,
) -> TalkBackendContext:
    return TalkBackendContext(
        camera_name="front",
        source_backend="rtsp",
        runtime_talk=TalkConfig(),
        camera_talk=camera_talk,
        fingerprint=fingerprint or CameraTalkFingerprint(),
    )


def _fake_detector(context: TalkBackendContext) -> TalkBackendDetection:
    if "fake_proprietary" in context.camera_talk.backends:
        return TalkBackendDetection(
            backend="fake_proprietary",
            confidence="explicit",
            reason="fake backend config present",
            safe_to_probe=True,
        )
    if context.fingerprint.manufacturer == "fake-vendor":
        return TalkBackendDetection(
            backend="fake_proprietary",
            confidence="high",
            reason="fake vendor fingerprint",
            safe_to_probe=True,
        )
    return TalkBackendDetection(
        backend="fake_proprietary",
        confidence="not_applicable",
        reason="no fake backend config or fingerprint",
        safe_to_probe=False,
    )


def _registry(
    *,
    calls: list[str],
    sessions: list[_FakeProprietaryTalkSession] | None = None,
    standard_capability: TalkCapabilityState = TalkCapabilityState.UNSUPPORTED,
) -> TalkBackendRegistry:
    session_store = sessions if sessions is not None else []
    registry = TalkBackendRegistry()
    registry.register(
        TalkBackendRegistration(
            name="onvif_rtsp_backchannel",
            config_model=_FakeConfig,
            factory=lambda config, context: _StandardTalkBackend(
                calls=calls,
                capability=standard_capability,
            ),
            priority=10,
            standards_based=True,
        )
    )
    registry.register(
        TalkBackendRegistration(
            name="fake_proprietary",
            config_model=_FakeConfig,
            factory=lambda config, context: _FakeProprietaryTalkBackend(
                config=config if isinstance(config, _FakeConfig) else _FakeConfig(),
                calls=calls,
                sessions=session_store,
            ),
            detector=_fake_detector,
            priority=20,
            standards_based=False,
        )
    )
    return registry


@pytest.mark.asyncio
async def test_explicit_fake_proprietary_backend_is_selected_without_onvif_probe() -> None:
    """Explicit proprietary backend selection should not fall back to ONVIF."""
    calls: list[str] = []
    sessions: list[_FakeProprietaryTalkSession] = []
    selector = TalkBackendSelector(
        registry=_registry(calls=calls, sessions=sessions),
        context=_context(
            CameraTalkConfig(
                backend="fake_proprietary",
                backends={"fake_proprietary": {"supported": True}},
            )
        ),
    )

    # Given: A camera explicitly configured for a registered fake proprietary backend
    # When: Probing and opening through the selector
    probe = await selector.probe()
    session = await selector.open_session(
        TalkSessionOpenRequest(session_id="tk_explicit", input=TalkInputFormat())
    )

    # Then: Only the fake backend is used and its selected codec is exposed
    assert probe.capability == TalkCapabilityState.SUPPORTED
    assert selector.backend == "fake_proprietary"
    assert session.selected_codec == "PCMA/8000"
    assert calls == ["fake:probe", "fake:open:tk_explicit"]


@pytest.mark.asyncio
async def test_auto_prefers_supported_onvif_before_fake_fingerprint() -> None:
    """Auto selection should keep ONVIF first when the standards backend works."""
    calls: list[str] = []
    selector = TalkBackendSelector(
        registry=_registry(
            calls=calls,
            standard_capability=TalkCapabilityState.SUPPORTED,
        ),
        context=_context(
            CameraTalkConfig(backend="auto"),
            fingerprint=CameraTalkFingerprint(manufacturer="fake-vendor"),
        ),
    )

    # Given: A safe fake fingerprint and a supported standards backend
    # When: Probing auto talk capability
    probe = await selector.probe()

    # Then: ONVIF wins and the proprietary backend is not probed
    assert probe.capability == TalkCapabilityState.SUPPORTED
    assert selector.backend == "onvif_rtsp_backchannel"
    assert calls == ["onvif:probe"]


@pytest.mark.asyncio
async def test_auto_selects_safe_fake_backend_after_onvif_is_unsupported() -> None:
    """Auto selection should fall through to safe proprietary candidates after ONVIF fails."""
    calls: list[str] = []
    selector = TalkBackendSelector(
        registry=_registry(calls=calls),
        context=_context(
            CameraTalkConfig(
                backend="auto",
                backends={"fake_proprietary": {"supported": True}},
            )
        ),
    )

    # Given: ONVIF is unsupported but fake proprietary config makes that backend safe to probe
    # When: Probing auto talk capability
    probe = await selector.probe()

    # Then: The fake backend is selected after the ONVIF probe fails
    assert probe.capability == TalkCapabilityState.SUPPORTED
    assert selector.backend == "fake_proprietary"
    assert calls == ["onvif:probe", "fake:probe"]


@pytest.mark.asyncio
async def test_auto_ignores_fake_backend_without_safe_config_or_fingerprint() -> None:
    """Auto selection should not probe unsafe proprietary candidates."""
    calls: list[str] = []
    selector = TalkBackendSelector(
        registry=_registry(calls=calls),
        context=_context(CameraTalkConfig(backend="auto")),
    )

    # Given: ONVIF is unsupported and the fake backend has no safe detector match
    # When: Probing auto talk capability
    probe = await selector.probe()

    # Then: The ONVIF unsupported result remains visible and fake is ignored
    assert probe.capability == TalkCapabilityState.UNSUPPORTED
    assert probe.refusal_reason == TalkRefusalReason.UNSUPPORTED_CAMERA
    assert selector.backend == "onvif_rtsp_backchannel"
    assert calls == ["onvif:probe"]


@pytest.mark.asyncio
async def test_fake_backend_session_receives_pcm_through_talk_manager() -> None:
    """A proprietary backend should work behind the existing TalkManager interface."""
    calls: list[str] = []
    sessions: list[_FakeProprietaryTalkSession] = []
    selector = TalkBackendSelector(
        registry=_registry(calls=calls, sessions=sessions),
        context=_context(
            CameraTalkConfig(
                backend="auto",
                backends={"fake_proprietary": {"supported": True}},
            )
        ),
    )
    manager = TalkManager(
        camera_name="front",
        enabled=True,
        supported_codecs=selector.supported_codecs,
        open_session_factory=selector.open_session,
        capability_probe_factory=selector.probe,
        max_session_s=60.0,
        idle_timeout_s=60.0,
    )
    frame = b"\x00" * TalkInputFormat().expected_bytes_per_frame

    # Given: A TalkManager wired to selector factories for a safe fake proprietary backend
    # When: Preparing, opening, writing, and stopping a talk session
    prepared = await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_fake"))
    session = await manager.open_session(TalkSessionOpenRequest(session_id="tk_fake"))
    await manager.write_pcm_frame("tk_fake", frame)
    stopped = await manager.stop_session("tk_fake")

    # Then: Existing manager lifecycle forwards PCM to the fake backend with no API/runtime changes
    assert prepared.accepted is True
    assert session is sessions[0]
    assert sessions[0].selected_codec == "PCMA/8000"
    assert sessions[0].frames == [frame]
    assert manager.status().selected_codec == "PCMA/8000"
    assert stopped is True
    assert sessions[0].closed is True
    assert calls == ["onvif:probe", "fake:probe", "fake:open:tk_fake"]
