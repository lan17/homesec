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
    TalkBackendConfigError,
    TalkBackendContext,
    TalkBackendDetection,
    TalkBackendOpenError,
    TalkBackendRegistration,
    TalkBackendRegistry,
)
from homesec.talk.selector import TalkBackendSelector


class _BackendConfig(BaseModel):
    marker: str = "default"


class _IntBackendConfig(BaseModel):
    marker: int


class _FakeSession:
    session_id = "tk_fake"
    camera_name = "cam"
    selected_codec = "PCMU/8000"

    async def write_pcm_frame(self, frame: bytes) -> None:
        _ = frame

    async def close(self) -> None:
        return None


class _FakeBackend:
    def __init__(
        self,
        name: str,
        calls: list[str],
        probe_result: TalkCapabilityProbeResult | None = None,
    ) -> None:
        self.name = name
        self._calls = calls
        self._probe_result = probe_result or TalkCapabilityProbeResult(
            capability=TalkCapabilityState.SUPPORTED
        )

    @property
    def supported_codecs(self) -> list[str]:
        return ["PCMU/8000"]

    async def probe(self) -> TalkCapabilityProbeResult:
        self._calls.append(f"{self.name}:probe")
        return self._probe_result

    async def open_session(self, request: TalkSessionOpenRequest) -> _FakeSession:
        self._calls.append(f"{self.name}:open:{request.session_id}")
        return _FakeSession()


class _PreparedProbeBackend(_FakeBackend):
    async def probe_for_session_open(self) -> TalkCapabilityProbeResult:
        self._calls.append(f"{self.name}:prepare_probe")
        return self._probe_result

    async def clear_prepared_probe(self) -> None:
        self._calls.append(f"{self.name}:clear_prepared_probe")


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
async def test_selector_clear_prepared_probe_delegates_to_selected_backend() -> None:
    """Selector should clear prepared state on the backend selected by auto probing."""
    # Given: A selector whose standards backend preserves state for session open
    calls: list[str] = []
    registry = TalkBackendRegistry()
    registry.register(
        TalkBackendRegistration(
            name="standard_backend",
            config_model=_BackendConfig,
            factory=lambda config, context: _PreparedProbeBackend("standard_backend", calls),
            priority=10,
            standards_based=True,
        )
    )
    selector = TalkBackendSelector(
        registry=registry,
        context=_context(CameraTalkConfig(backend="auto")),
    )

    # When: A prepare-time probe selects the backend and cleanup is requested
    probe = await selector.probe_for_session_open()
    await selector.clear_prepared_probe()

    # Then: Cleanup is routed to the selected backend instance
    assert probe.capability == TalkCapabilityState.SUPPORTED
    assert calls == ["standard_backend:prepare_probe", "standard_backend:clear_prepared_probe"]


@pytest.mark.asyncio
async def test_selector_auto_uses_standards_backend_before_safe_detector() -> None:
    """Backend auto mode should try standards before proprietary detector matches."""
    # Given: A selector in auto mode with a supported standard backend and safe vendor detector
    calls: list[str] = []
    registry = _registry(calls)
    registry.register(
        TalkBackendRegistration(
            name="detected_vendor",
            config_model=_BackendConfig,
            factory=lambda config, context: _FakeBackend("detected_vendor", calls),
            detector=lambda context: pytest.fail(
                "proprietary detector should wait until standards fail"
            ),
            priority=200,
            standards_based=False,
        )
    )
    selector = TalkBackendSelector(
        registry=registry,
        context=_context(CameraTalkConfig(backend="auto")),
    )

    # When: Probing and opening a talk session
    probe = await selector.probe()
    session = await selector.open_session(
        TalkSessionOpenRequest(session_id="tk_detected", input=TalkInputFormat())
    )

    # Then: The standards backend wins and the detected vendor is not probed
    assert probe.capability == TalkCapabilityState.SUPPORTED
    assert session.selected_codec == "PCMU/8000"
    assert calls == ["standard_backend:probe", "standard_backend:open:tk_detected"]


@pytest.mark.asyncio
async def test_selector_auto_continues_after_candidate_config_error() -> None:
    """Backend auto mode should not let one invalid candidate block a later supported one."""
    # Given: The first standards backend has invalid config but a later standards backend works
    calls: list[str] = []
    registry = TalkBackendRegistry()
    registry.register(
        TalkBackendRegistration(
            name="bad_standard_backend",
            config_model=_IntBackendConfig,
            factory=lambda config, context: _FakeBackend("bad_standard_backend", calls),
            priority=1,
            standards_based=True,
        )
    )
    registry.register(
        TalkBackendRegistration(
            name="good_standard_backend",
            config_model=_BackendConfig,
            factory=lambda config, context: _FakeBackend("good_standard_backend", calls),
            priority=2,
            standards_based=True,
        )
    )
    selector = TalkBackendSelector(
        registry=registry,
        context=_context(CameraTalkConfig(backend="auto")),
    )

    # When: Probing and opening through auto selection
    probe = await selector.probe()
    session = await selector.open_session(
        TalkSessionOpenRequest(session_id="tk_config", input=TalkInputFormat())
    )

    # Then: The valid later standards backend is selected
    assert probe.capability == TalkCapabilityState.SUPPORTED
    assert selector.backend == "good_standard_backend"
    assert session.selected_codec == "PCMU/8000"
    assert calls == ["good_standard_backend:probe", "good_standard_backend:open:tk_config"]


@pytest.mark.asyncio
async def test_selector_auto_falls_back_to_safe_detector_after_standard_unsupported() -> None:
    """Backend auto mode should use a safe proprietary backend after standards fail."""
    # Given: A selector in auto mode with unsupported standard talk and a safe vendor detector
    calls: list[str] = []
    registry = TalkBackendRegistry()
    registry.register(
        TalkBackendRegistration(
            name="standard_backend",
            config_model=_BackendConfig,
            factory=lambda config, context: _FakeBackend(
                "standard_backend",
                calls,
                probe_result=TalkCapabilityProbeResult(
                    capability=TalkCapabilityState.UNSUPPORTED,
                    refusal_reason=TalkRefusalReason.UNSUPPORTED_CAMERA,
                    message="standard backend unsupported",
                ),
            ),
            priority=10,
            standards_based=True,
        )
    )
    registry.register(
        TalkBackendRegistration(
            name="detected_vendor",
            config_model=_BackendConfig,
            factory=lambda config, context: _FakeBackend("detected_vendor", calls),
            detector=lambda context: TalkBackendDetection(
                backend="detected_vendor",
                confidence="high",
                reason="camera fingerprint matched detected vendor",
                safe_to_probe=True,
            ),
            priority=200,
            standards_based=False,
        )
    )
    selector = TalkBackendSelector(
        registry=registry,
        context=_context(CameraTalkConfig(backend="auto")),
    )

    # When: Probing and opening a talk session
    probe = await selector.probe()
    session = await selector.open_session(
        TalkSessionOpenRequest(session_id="tk_detected", input=TalkInputFormat())
    )

    # Then: The selector falls through from standards to the safe vendor backend
    assert probe.capability == TalkCapabilityState.SUPPORTED
    assert selector.backend == "detected_vendor"
    assert session.selected_codec == "PCMU/8000"
    assert calls == [
        "standard_backend:probe",
        "detected_vendor:probe",
        "detected_vendor:open:tk_detected",
    ]


@pytest.mark.asyncio
async def test_selector_auto_reports_safe_detector_failure_after_standard_unsupported() -> None:
    """Backend auto mode should surface actionable fallback failures when all candidates fail."""
    # Given: A standards backend is unsupported and a safe vendor detector hits auth failure
    calls: list[str] = []
    registry = TalkBackendRegistry()
    registry.register(
        TalkBackendRegistration(
            name="standard_backend",
            config_model=_BackendConfig,
            factory=lambda config, context: _FakeBackend(
                "standard_backend",
                calls,
                probe_result=TalkCapabilityProbeResult(
                    capability=TalkCapabilityState.UNSUPPORTED,
                    refusal_reason=TalkRefusalReason.UNSUPPORTED_CAMERA,
                    message="standard backend unsupported",
                ),
            ),
            priority=10,
            standards_based=True,
        )
    )
    registry.register(
        TalkBackendRegistration(
            name="detected_vendor",
            config_model=_BackendConfig,
            factory=lambda config, context: _FakeBackend(
                "detected_vendor",
                calls,
                probe_result=TalkCapabilityProbeResult(
                    capability=TalkCapabilityState.ERROR,
                    refusal_reason=TalkRefusalReason.TALK_AUTH_FAILED,
                    message="fake vendor auth failed",
                ),
            ),
            detector=lambda context: TalkBackendDetection(
                backend="detected_vendor",
                confidence="high",
                reason="camera fingerprint matched detected vendor",
                safe_to_probe=True,
            ),
            priority=200,
            standards_based=False,
        )
    )
    selector = TalkBackendSelector(
        registry=registry,
        context=_context(CameraTalkConfig(backend="auto")),
    )

    # When: Probing all auto candidates
    probe = await selector.probe()

    # Then: The safe vendor auth failure is reported instead of hiding behind ONVIF unsupported
    assert probe.capability == TalkCapabilityState.ERROR
    assert probe.refusal_reason == TalkRefusalReason.TALK_AUTH_FAILED
    assert probe.message == "fake vendor auth failed"
    assert selector.backend == "detected_vendor"
    assert calls == ["standard_backend:probe", "detected_vendor:probe"]


@pytest.mark.asyncio
async def test_selector_auto_reports_detector_failure_after_standard_unsupported() -> None:
    """Backend auto mode should isolate proprietary detector failures to fallback probing."""
    # Given: Standards probing fails and a proprietary detector raises before selection
    calls: list[str] = []
    registry = TalkBackendRegistry()
    registry.register(
        TalkBackendRegistration(
            name="standard_backend",
            config_model=_BackendConfig,
            factory=lambda config, context: _FakeBackend(
                "standard_backend",
                calls,
                probe_result=TalkCapabilityProbeResult(
                    capability=TalkCapabilityState.UNSUPPORTED,
                    refusal_reason=TalkRefusalReason.UNSUPPORTED_CAMERA,
                    message="standard backend unsupported",
                ),
            ),
            priority=10,
            standards_based=True,
        )
    )
    registry.register(
        TalkBackendRegistration(
            name="detected_vendor",
            config_model=_BackendConfig,
            factory=lambda config, context: _FakeBackend("detected_vendor", calls),
            detector=lambda context: (_ for _ in ()).throw(RuntimeError("detector boom")),
            priority=200,
            standards_based=False,
        )
    )
    selector = TalkBackendSelector(
        registry=registry,
        context=_context(CameraTalkConfig(backend="auto")),
    )

    # When: Probing through auto selection
    probe = await selector.probe()

    # Then: The detector failure is reported without raising or skipping standards probing
    assert probe.capability == TalkCapabilityState.ERROR
    assert probe.refusal_reason == TalkRefusalReason.RUNTIME_UNAVAILABLE
    assert probe.message == "Talk backend 'detected_vendor' detector failed"
    assert selector.backend == "detected_vendor"
    assert calls == ["standard_backend:probe"]


def test_selector_auto_supported_codecs_does_not_build_proprietary_fallback() -> None:
    """Startup codec metadata should not instantiate proprietary fallback backends."""
    # Given: A safe proprietary detector whose backend factory must wait for standards failure
    calls: list[str] = []
    registry = _registry(calls)
    registry.register(
        TalkBackendRegistration(
            name="detected_vendor",
            config_model=_BackendConfig,
            factory=lambda config, context: pytest.fail(
                "proprietary fallback should not build for startup codec metadata"
            ),
            detector=lambda context: TalkBackendDetection(
                backend="detected_vendor",
                confidence="high",
                reason="camera fingerprint matched detected vendor",
                safe_to_probe=True,
            ),
            priority=200,
            standards_based=False,
        )
    )
    selector = TalkBackendSelector(
        registry=registry,
        context=_context(CameraTalkConfig(backend="auto")),
    )

    # When: Source startup asks for supported codecs before any capability probe
    codecs = selector.supported_codecs

    # Then: Only standards-based codec metadata is built
    assert codecs == ["PCMU/8000"]
    assert selector.backend is None
    assert calls == []


@pytest.mark.asyncio
async def test_selector_auto_ignores_detector_that_is_not_safe_to_probe() -> None:
    """Backend auto mode should not auto-probe unsafe detector matches."""
    # Given: A selector in auto mode with a vendor detector that is not safe to probe
    calls: list[str] = []
    registry = TalkBackendRegistry()
    registry.register(
        TalkBackendRegistration(
            name="standard_backend",
            config_model=_BackendConfig,
            factory=lambda config, context: _FakeBackend(
                "standard_backend",
                calls,
                probe_result=TalkCapabilityProbeResult(
                    capability=TalkCapabilityState.UNSUPPORTED,
                    refusal_reason=TalkRefusalReason.UNSUPPORTED_CAMERA,
                    message="standard backend unsupported",
                ),
            ),
            priority=10,
            standards_based=True,
        )
    )
    registry.register(
        TalkBackendRegistration(
            name="unsafe_vendor",
            config_model=_BackendConfig,
            factory=lambda config, context: _FakeBackend("unsafe_vendor", calls),
            detector=lambda context: TalkBackendDetection(
                backend="unsafe_vendor",
                confidence="high",
                reason="camera fingerprint matched but probing is not safe",
                safe_to_probe=False,
            ),
            priority=1,
            standards_based=False,
        )
    )
    selector = TalkBackendSelector(
        registry=registry,
        context=_context(CameraTalkConfig(backend="auto")),
    )

    # When: Probing through auto selection
    probe = await selector.probe()

    # Then: Selection preserves the standard result instead of probing the unsafe vendor
    assert probe.capability == TalkCapabilityState.UNSUPPORTED
    assert probe.refusal_reason == TalkRefusalReason.UNSUPPORTED_CAMERA
    assert selector.backend == "standard_backend"
    assert calls == ["standard_backend:probe"]


@pytest.mark.asyncio
async def test_selector_auto_refuses_vendor_only_registry_without_safe_detector() -> None:
    """Backend auto mode should not probe proprietary backends without safe detection."""
    # Given: A registry with only a vendor backend and no safe detector match
    calls: list[str] = []
    registry = TalkBackendRegistry()
    registry.register(
        TalkBackendRegistration(
            name="vendor_backend",
            config_model=_BackendConfig,
            factory=lambda config, context: _FakeBackend("vendor_backend", calls),
            priority=1,
            standards_based=False,
        )
    )
    selector = TalkBackendSelector(
        registry=registry,
        context=_context(CameraTalkConfig(backend="auto")),
    )

    # When: Probing through auto selection
    probe = await selector.probe()

    # Then: Selection returns a runtime error without probing the vendor backend
    assert probe.capability == TalkCapabilityState.ERROR
    assert probe.refusal_reason == TalkRefusalReason.RUNTIME_UNAVAILABLE
    assert probe.message == "No standards-based talk backends are registered"
    assert selector.supported_codecs == []
    assert calls == []


@pytest.mark.asyncio
async def test_selector_auto_refuses_safe_vendor_when_no_standard_backend_is_registered() -> None:
    """Safe proprietary detections should not replace the standards-first prerequisite."""
    # Given: A registry with only a safe vendor detector and no standards backend
    calls: list[str] = []
    registry = TalkBackendRegistry()
    registry.register(
        TalkBackendRegistration(
            name="vendor_backend",
            config_model=_BackendConfig,
            factory=lambda config, context: _FakeBackend("vendor_backend", calls),
            detector=lambda context: TalkBackendDetection(
                backend="vendor_backend",
                confidence="high",
                reason="camera fingerprint matched vendor",
                safe_to_probe=True,
            ),
            priority=1,
            standards_based=False,
        )
    )
    selector = TalkBackendSelector(
        registry=registry,
        context=_context(CameraTalkConfig(backend="auto")),
    )

    # When: Probing through auto selection
    probe = await selector.probe()

    # Then: Auto selection refuses without probing proprietary-only candidates
    assert probe.capability == TalkCapabilityState.ERROR
    assert probe.refusal_reason == TalkRefusalReason.RUNTIME_UNAVAILABLE
    assert probe.message == "No standards-based talk backends are registered"
    assert selector.backend is None
    assert calls == []


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

    # Then: Selection reports a config error without probing fallback backends
    assert probe.capability == TalkCapabilityState.CONFIG_ERROR
    assert probe.refusal_reason == TalkRefusalReason.TALK_CONFIG_ERROR
    assert probe.message == "Talk backend 'missing_vendor' is not registered in this runtime"
    assert selector.supported_codecs == []
    assert calls == []

    # When: Opening a session with the same missing explicit backend
    # Then: The selector raises the same refusal reason for TalkManager mapping
    with pytest.raises(TalkBackendOpenError) as exc_info:
        await selector.open_session(
            TalkSessionOpenRequest(session_id="tk_missing", input=TalkInputFormat())
        )
    assert exc_info.value.reason == TalkRefusalReason.TALK_CONFIG_ERROR


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


@pytest.mark.asyncio
async def test_selector_reports_stable_public_message_for_invalid_backend_config() -> None:
    """Invalid backend config should not expose raw validation text in status errors."""
    # Given: A registered backend whose config validation would include rejected input
    registry = TalkBackendRegistry()
    registry.register(
        TalkBackendRegistration(
            name="vendor_backend",
            config_model=_IntBackendConfig,
            factory=lambda config, context: _FakeBackend("vendor_backend", []),
        )
    )
    selector = TalkBackendSelector(
        registry=registry,
        context=_context(
            CameraTalkConfig(
                backend="vendor_backend",
                config={"marker": "secret-config-value"},
            )
        ),
    )

    # When: Probing the selected backend
    probe = await selector.probe()

    # Then: The public error is stable and does not include raw Pydantic input text
    assert probe.capability == TalkCapabilityState.CONFIG_ERROR
    assert probe.refusal_reason == TalkRefusalReason.TALK_CONFIG_ERROR
    assert probe.message == "Talk backend 'vendor_backend' config is invalid"
    assert selector.backend_reason == "Talk backend 'vendor_backend' config is invalid"
    assert "secret-config-value" not in (probe.message or "")


@pytest.mark.asyncio
async def test_selector_preserves_safe_structured_config_error_message() -> None:
    """Structured config errors should let backends expose safe public messages."""

    # Given: A backend config factory reports a safe public config error
    def _raise_structured_config_error(
        raw_config: dict[str, object] | BaseModel,
        context: TalkBackendContext,
    ) -> BaseModel:
        _ = raw_config, context
        raise TalkBackendConfigError(
            "Required talk backend environment variable is not set: OFFICE_TAPO_SHA256"
        )

    registry = TalkBackendRegistry()
    registry.register(
        TalkBackendRegistration(
            name="vendor_backend",
            config_model=_BackendConfig,
            config_factory=_raise_structured_config_error,
            factory=lambda config, context: _FakeBackend("vendor_backend", []),
        )
    )
    selector = TalkBackendSelector(
        registry=registry,
        context=_context(CameraTalkConfig(backend="vendor_backend")),
    )

    # When: Probing the selected backend
    probe = await selector.probe()

    # Then: The backend's safe public error is preserved
    assert probe.capability == TalkCapabilityState.CONFIG_ERROR
    assert probe.refusal_reason == TalkRefusalReason.TALK_CONFIG_ERROR
    assert (
        probe.message == "Required talk backend environment variable is not set: OFFICE_TAPO_SHA256"
    )
    assert selector.backend_reason == probe.message


@pytest.mark.asyncio
async def test_selector_drops_unsafe_structured_config_error_message() -> None:
    """Structured config errors should still pass through public sanitization."""

    # Given: A backend config factory mistakenly includes an unsafe URL in its public error
    def _raise_unsafe_structured_config_error(
        raw_config: dict[str, object] | BaseModel,
        context: TalkBackendContext,
    ) -> BaseModel:
        _ = raw_config, context
        raise TalkBackendConfigError(
            "Selected rtsp://alice:secret@camera.local/talk for backend config"
        )

    registry = TalkBackendRegistry()
    registry.register(
        TalkBackendRegistration(
            name="vendor_backend",
            config_model=_BackendConfig,
            config_factory=_raise_unsafe_structured_config_error,
            factory=lambda config, context: _FakeBackend("vendor_backend", []),
        )
    )
    selector = TalkBackendSelector(
        registry=registry,
        context=_context(CameraTalkConfig(backend="vendor_backend")),
    )

    # When: Probing the selected backend
    probe = await selector.probe()

    # Then: Unsafe structured detail falls back to the generic backend config error
    assert probe.capability == TalkCapabilityState.CONFIG_ERROR
    assert probe.refusal_reason == TalkRefusalReason.TALK_CONFIG_ERROR
    assert probe.message == "Talk backend 'vendor_backend' config is invalid"
    assert selector.backend_reason == "Talk backend 'vendor_backend' config is invalid"
    assert "rtsp://alice:secret@camera.local/talk" not in (probe.message or "")


@pytest.mark.asyncio
async def test_selector_does_not_parse_plain_value_error_missing_env_strings() -> None:
    """Generic selection should not parse backend-specific ValueError messages."""

    # Given: A backend config factory raises an RTSP-shaped plain ValueError
    def _raise_plain_missing_env(
        raw_config: dict[str, object] | BaseModel,
        context: TalkBackendContext,
    ) -> BaseModel:
        _ = raw_config, context
        raise ValueError("RTSP URL environment variable is not set: SAFE_TALK_RTSP_URL")

    registry = TalkBackendRegistry()
    registry.register(
        TalkBackendRegistration(
            name="vendor_backend",
            config_model=_BackendConfig,
            config_factory=_raise_plain_missing_env,
            factory=lambda config, context: _FakeBackend("vendor_backend", []),
        )
    )
    selector = TalkBackendSelector(
        registry=registry,
        context=_context(CameraTalkConfig(backend="vendor_backend")),
    )

    # When: Probing the selected backend
    probe = await selector.probe()

    # Then: The generic selector does not preserve backend-specific string details
    assert probe.capability == TalkCapabilityState.CONFIG_ERROR
    assert probe.refusal_reason == TalkRefusalReason.TALK_CONFIG_ERROR
    assert probe.message == "Talk backend 'vendor_backend' config is invalid"
    assert selector.backend_reason == "Talk backend 'vendor_backend' config is invalid"
    assert "SAFE_TALK_RTSP_URL" not in (probe.message or "")
