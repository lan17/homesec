"""Tests for backend-agnostic talk registry primitives."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from homesec.models.config import CameraTalkConfig, TalkConfig
from homesec.models.talk import (
    TalkCapabilityProbeResult,
    TalkCapabilityState,
    TalkSessionOpenRequest,
)
from homesec.sources.rtsp.talk.backend import build_rtsp_talk_backend_registry
from homesec.talk.backends import (
    CameraTalkFingerprint,
    TalkBackendContext,
    TalkBackendDetection,
    TalkBackendRegistration,
    TalkBackendRegistry,
    backend_config_for,
    model_validate_backend_config,
)
from homesec.talk.registry import build_default_talk_backend_registry


class _BackendConfig(BaseModel):
    """Fake backend config for registry tests."""

    enabled: bool = True


class _OtherBackendConfig(BaseModel):
    """Second fake backend config for ordering tests."""

    enabled: bool = True


class _FakeSession:
    session_id = "tk_fake"
    camera_name = "front"
    selected_codec = "PCMU/8000"

    async def write_pcm_frame(self, frame: bytes) -> None:
        _ = frame

    async def close(self) -> None:
        return None


class _FakeBackend:
    name = "fake_backend"

    @property
    def supported_codecs(self) -> list[str]:
        return ["PCMU/8000"]

    async def probe(self) -> TalkCapabilityProbeResult:
        return TalkCapabilityProbeResult(capability=TalkCapabilityState.SUPPORTED)

    async def open_session(self, request: TalkSessionOpenRequest) -> _FakeSession:
        _ = request
        return _FakeSession()


def _registration(
    name: str,
    *,
    priority: int = 100,
    standards_based: bool = False,
) -> TalkBackendRegistration:
    return TalkBackendRegistration(
        name=name,
        config_model=_BackendConfig,
        factory=lambda config, context: _FakeBackend(),
        priority=priority,
        standards_based=standards_based,
    )


def test_registry_rejects_duplicate_backend_names() -> None:
    """Talk backend registry should reject duplicate names."""
    # Given: A registry with one backend registration
    registry = TalkBackendRegistry()
    registry.register(_registration("fake_backend"))

    # When: Registering the same backend again
    # Then: The registry fails clearly
    with pytest.raises(ValueError, match="fake_backend"):
        registry.register(_registration("FAKE_BACKEND"))


def test_registry_rejects_unsafe_backend_names() -> None:
    """Talk backend registration names should be safe public identifiers."""
    # Given: A backend registration whose name is not a safe identifier
    registry = TalkBackendRegistry()

    # When: Registering the unsafe backend
    # Then: The registry rejects it before diagnostics can expose the name
    with pytest.raises(ValueError, match="talk backend names"):
        registry.register(_registration("rtsp://admin:secret@example.local/stream1"))


def test_registry_orders_standards_backends_first() -> None:
    """Talk backend registry should return deterministic standards-first order."""
    # Given: Proprietary and standards-based backend registrations
    registry = TalkBackendRegistry()
    registry.register(_registration("vendor_backend", priority=1, standards_based=False))
    registry.register(_registration("onvif_rtsp_backchannel", priority=50, standards_based=True))
    registry.register(_registration("other_standard", priority=10, standards_based=True))

    # When: Reading the standards-first registry order
    names = registry.names()

    # Then: Standards-based registrations sort before proprietary candidates
    assert names == ("other_standard", "onvif_rtsp_backchannel", "vendor_backend")


def test_default_talk_backend_registry_contains_onvif_backend() -> None:
    """Default talk backend registry should include the built-in ONVIF backend."""
    # Given: The built-in HomeSec talk backend registry
    # When: Reading registered backend names
    registry = build_default_talk_backend_registry()

    # Then: ONVIF remains the standards-based default backend
    assert registry.names() == ("onvif_rtsp_backchannel",)
    registration = registry.get("onvif_rtsp_backchannel")
    assert registration is not None
    assert registration.standards_based is True


def test_rtsp_talk_backend_registry_alias_returns_default_registry() -> None:
    """Legacy RTSP registry helper should remain a compatibility alias."""
    # Given: The deprecated RTSP-named registry helper
    # When: Building a registry through the alias and the generic helper
    legacy_registry = build_rtsp_talk_backend_registry()
    default_registry = build_default_talk_backend_registry()

    # Then: Both expose the same built-in backend names
    assert legacy_registry.names() == default_registry.names()


def test_backend_context_resolves_env_and_redaction_through_injected_helpers() -> None:
    """Talk backend context should keep env and redaction helpers explicit."""
    # Given: Backend context with injected helpers
    context = TalkBackendContext(
        camera_name="front",
        source_backend="rtsp",
        runtime_talk=TalkConfig(),
        camera_talk=CameraTalkConfig(),
        fingerprint=CameraTalkFingerprint(manufacturer="fake-vendor"),
        resolve_env=lambda name: "secret-value" if name == "SECRET_ENV" else None,
        redact=lambda value: value.replace("secret-value", "***"),
    )

    # When: A backend resolves an env var and redacts a diagnostic value
    env_value = context.env_value("SECRET_ENV")
    diagnostic = context.redacted(f"value={env_value}")

    # Then: Resolution and redaction stay under injected caller control
    assert env_value == "secret-value"
    assert diagnostic == "value=***"


def test_detection_preserves_safe_probe_metadata() -> None:
    """Talk backend detection should carry safe auto-probe metadata."""
    # Given: A strong camera fingerprint for a backend detector
    # When: Building a detector result that marks the backend as safe to probe
    detection = TalkBackendDetection(
        backend="fake_backend",
        confidence="high",
        reason="fake vendor fingerprint",
        safe_to_probe=True,
        requires_credentials=True,
    )

    # Then: Selection code can decide whether probing is allowed
    assert detection.backend == "fake_backend"
    assert detection.confidence == "high"
    assert detection.safe_to_probe is True
    assert detection.requires_credentials is True


def test_backend_config_for_preserves_compatibility_aliases() -> None:
    """Backend config lookup should preserve legacy and new config shapes."""
    # Given: Camera talk config with explicit and backend-map entries
    camera_talk = CameraTalkConfig(
        backend="auto",
        config={"preferred_codecs": ["PCMA/8000"]},
        backends={"fake_backend": {"enabled": False}},
    )

    # When: Reading config for ONVIF and a future backend
    onvif_config = backend_config_for(camera_talk, "onvif_rtsp_backchannel")
    fake_config = backend_config_for(camera_talk, "fake_backend")

    # Then: ONVIF gets the legacy alias and fake backend gets its own block
    assert onvif_config == {"preferred_codecs": ["PCMA/8000"]}
    assert fake_config == {"enabled": False}


def test_model_validate_backend_config_uses_registration_model() -> None:
    """Backend config validation should use the selected registration's config model."""
    # Given: A backend registration with a typed config model
    registration = TalkBackendRegistration(
        name="other_backend",
        config_model=_OtherBackendConfig,
        factory=lambda config, context: _FakeBackend(),
    )

    # When: Validating raw backend config through the registration
    config = model_validate_backend_config(registration, {"enabled": False})

    # Then: The selected backend's model owns validation
    assert isinstance(config, _OtherBackendConfig)
    assert config.enabled is False
